import argparse, os
import torch.nn as nn

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F

##import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device, ClipLoss2
import datetime
run_id = datetime.datetime.now().strftime("%m%d_%H%M%S")
# device = get_device('auto')
from omegaconf import DictConfig
import math
def compute_sample_reliability_from_phase_stats(
    amp_phase_stats: dict,
    detach: bool = True,
):
    """
    根据相位圆方差 phase_circ_var 计算样本级可靠度:
        r_i = 1 - mean_c(CircVar_ic)
    amp_phase_stats["phase_circ_var"]: [B, C]
    return:
        reliability: [B], in [0, 1]
    """
    if amp_phase_stats is None:
        return None

    phase_circ_var = amp_phase_stats["phase_circ_var"]   # [B, C]
    reliability = 1.0 - phase_circ_var.mean(dim=-1)      # [B]
    reliability = reliability.clamp(0.0, 1.0)

    if detach:
        reliability = reliability.detach()

    return reliability
class AdaptiveTemperature(nn.Module):
    """
    tau_i = softplus(a - b * r_i) + tau_min
    r_i 越大 => 样本越可靠 => tau_i 越小 => logits 越 sharp
    """
    def __init__(self, init_a=-2.0, init_b=1.0, tau_min=0.01):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.tau_min = tau_min
        self.softplus = nn.Softplus()

    def forward(self, reliability: torch.Tensor):
        """
        reliability: [B]
        return:
            tau: [B]
        """
        tau = self.softplus(self.a - self.b * reliability) + self.tau_min
        return tau
class AdaptiveClipLoss(nn.Module):
    """
    支持 per-sample temperature 的对称 CLIP loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, eeg_z: torch.Tensor, img_z: torch.Tensor, tau: torch.Tensor):
        """
        eeg_z: [B, D]
        img_z: [B, D]
        tau  : [B]
        return:
            loss_eeg: [B]
            loss_img: [B]
            logits_eeg: [B, B]
        """
        eeg_z = F.normalize(eeg_z, dim=-1)
        img_z = F.normalize(img_z, dim=-1)

        sim = eeg_z @ img_z.T   # [B, B]
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        tau = tau.view(B, 1)    # row-wise temperature

        logits_eeg = sim / tau      # eeg -> img
        logits_img = sim.T / tau    # img -> eeg

        loss_eeg = F.cross_entropy(logits_eeg, labels, reduction='none')
        loss_img = F.cross_entropy(logits_img, labels, reduction='none')

        return loss_eeg, loss_img, logits_eeg
class AmpPhaseDistributionConditioner(nn.Module):
    """
    从 EEG 的频域幅值/相位中提取分布统计：
      - amp mean
      - amp std
      - phase circular mean (via cos/sin mean)
      - phase circular variance

    然后编码为 cond 向量。
    """
    def __init__(
        self,
        n_channels=17,
        n_times=250,
        cond_dim=128,
        hidden_dim=128,
        stats_reduce="channel",   # "global" or "channel"
        use_layernorm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_freq = n_times // 2 + 1
        self.stats_reduce = stats_reduce
        self.eps = eps

        if stats_reduce == "global":
            # amp_mean, amp_std, phase_cos_mean, phase_sin_mean, circ_var => 5
            in_dim = 5
        elif stats_reduce == "channel":
            # 每个 channel 一组 5 维统计
            in_dim = n_channels * 5
        else:
            raise ValueError(f"Unknown stats_reduce: {stats_reduce}")

        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(in_dim))
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        x: [B,1,C,T] or [B,C,T]
        return:
            cond: [B, cond_dim]
            stats_dict: dict of summary stats
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # [B,C,T]
        assert x.dim() == 3, f"Expected [B,C,T], got {x.shape}"

        Xf = torch.fft.rfft(x, dim=-1)      # [B,C,F]
        amp = torch.abs(Xf)                 # [B,C,F]
        phase = torch.angle(Xf)             # [B,C,F]

        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # 幅值统计
        amp_mean = amp.mean(dim=-1)                         # [B,C]
        amp_std = amp.std(dim=-1, unbiased=False)          # [B,C]

        # 相位圆统计
        cos_mean = cos_p.mean(dim=-1)                      # [B,C]
        sin_mean = sin_p.mean(dim=-1)                      # [B,C]
        R = torch.sqrt(cos_mean.pow(2) + sin_mean.pow(2) + self.eps)  # [B,C]
        circ_var = 1.0 - R                                 # [B,C]

        if self.stats_reduce == "global":
            feat = torch.stack([
                amp_mean.mean(dim=-1),     # [B]
                amp_std.mean(dim=-1),
                cos_mean.mean(dim=-1),
                sin_mean.mean(dim=-1),
                circ_var.mean(dim=-1),
            ], dim=-1)                     # [B,5]
        else:  # channel
            feat = torch.cat([
                amp_mean,                  # [B,C]
                amp_std,                   # [B,C]
                cos_mean,                  # [B,C]
                sin_mean,                  # [B,C]
                circ_var,                  # [B,C]
            ], dim=-1)                     # [B,5C]

        cond = self.net(feat)

        stats_dict = {
            "amp_mean": amp_mean,
            "amp_std": amp_std,
            "phase_cos_mean": cos_mean,
            "phase_sin_mean": sin_mean,
            "phase_circ_var": circ_var,
        }
        return cond, stats_dict
    
class FiLMModulator(nn.Module):
    """
    用 cond 生成 gamma/beta，调制 eeg embedding:
        z' = gamma * z + beta
    """
    def __init__(self, z_dim, cond_dim=128, hidden_dim=256):
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.to_conf = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        conf = torch.sigmoid(self.to_conf(cond))   # [B,1]

        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        # confidence-controlled FiLM
        out = (1.0 - conf) * z + conf * (gamma * z + beta)
        return out, gamma, beta, conf
    
class CCAStyleAdapter(nn.Module):
    def __init__(self, dim, rank=256, dropout=0.1, residual=True, use_ln=True):
        super().__init__()
        self.proj_down = nn.Linear(dim, rank, bias=False)
        self.proj_up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim, eps=1e-6) if use_ln else nn.Identity()

    def forward(self, x, return_lowrank=False):
        """
        x: [B, D]
        return:
            out: [B, D]
            lowrank(optional): [B, rank]
        """
        lowrank = self.proj_down(x)                 
        out = self.proj_up(self.dropout(lowrank))  

        if self.residual:
            out = out + x

        out = self.ln(out)

        if return_lowrank:
            return out, lowrank
        return out


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cca_correlation_loss(z_eeg_low: torch.Tensor,
                         z_img_low: torch.Tensor,
                         eps: float = 1e-8,
                         lambda_offdiag: float = 0.005) -> torch.Tensor:
    z1 = z_eeg_low.float()
    z2 = z_img_low.float()

    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0, unbiased=False) + eps)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0, unbiased=False) + eps)

    B = z1.shape[0]
    c = (z1.T @ z2) / float(B)

    on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()

    return on_diag + lambda_offdiag * off_diag
class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectorLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
class ResidualMLPAdapter(nn.Module):
    """
    小型图像适配头：
    输入固定 img feature [B, D]
    输出适配后的 img feature [B, D]
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1, use_ln=True, residual=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim

        layers = []
        if use_ln:
            layers.append(nn.LayerNorm(dim))
        layers.extend([
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        ])
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = out + x
        return out
def count_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def _format_num(n: int) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}K"
    return str(n)

def print_module_tree(module: torch.nn.Module, name="model", max_depth=4):
    """
    打印模块树（结构层级），用于快速看网络由哪些 block 组成。
    """
    print(f"\n===== Module Tree: {name} (max_depth={max_depth}) =====")
    def _rec(m, prefix, depth):
        if depth > max_depth:
            return
        for k, v in m.named_children():
            cls = v.__class__.__name__
            tot, trn = count_params(v)
            print(f"{prefix}- {k}: {cls} | params={_format_num(tot)} trainable={_format_num(trn)}")
            _rec(v, prefix + "  ", depth + 1)
    _rec(module, "", 1)

@torch.no_grad()
def infer_one_pass(pl_model: pl.LightningModule, batch: dict, device: torch.device):
    """
    用一个 batch 跑 forward，检查关键张量维度是否符合预期。
    """
    pl_model.eval()
    # batch 里可能有 numpy/list 等，尽量转 torch 并搬到 device
    b = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            b[k] = v.to(device)
        else:
            b[k] = v  # idx/label 可能是 tensor，其他保持原样

    outputs = pl_model(b)
    eeg_z, img_z, loss = outputs[0], outputs[1], outputs[2]
    print("\n===== One-pass sanity check =====")
    print(f"eeg_z: {tuple(eeg_z.shape)}  dtype={eeg_z.dtype}  device={eeg_z.device}")
    print(f"img_z: {tuple(img_z.shape)}  dtype={img_z.dtype}  device={img_z.device}")
    if torch.is_tensor(loss):
        print(f"loss : {loss.item():.6f}")
    else:
        print(f"loss : {loss}")

def register_shape_hooks(module: torch.nn.Module, focus_keys=None):
    """
    注册 forward hook，打印各层输出 shape。
    focus_keys: 只打印名字包含这些子串的层（可选），避免输出过多。
    """
    hooks = []

    def _need(name: str) -> bool:
        if focus_keys is None:
            return True
        return any(k in name for k in focus_keys)

    def _hook(name):
        def fn(m, inp, out):
            def shape(x):
                if torch.is_tensor(x): return tuple(x.shape)
                if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
                    return [tuple(t.shape) for t in x]
                return str(type(x))
            if _need(name):
                in_shapes = []
                for x in inp:
                    in_shapes.append(shape(x))
                print(f"[HOOK] {name:<60} in={in_shapes}  out={shape(out)}")
        return fn

    for name, m in module.named_modules():
        # 只 hook 叶子层，避免重复信息
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(_hook(name)))

    return hooks

def print_model_report(pl_model: pl.LightningModule, sample_batch: dict, device=None):
    """
    一键输出：参数量、模块树、(可选)hook形状、一次forward sanity。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl_model = pl_model.to(device)

    print("\n================= MODEL REPORT =================")
    tot, trn = count_params(pl_model)
    print(f"[PLModel] total params: {_format_num(tot)} | trainable: {_format_num(trn)}")
    if hasattr(pl_model, "img_adapter"):
        tot_i, trn_i = count_params(pl_model.img_adapter)
        print(f"[img_adapter] total params: {_format_num(tot_i)} | trainable: {_format_num(trn_i)}")
        print_module_tree(pl_model.img_adapter, name="img_adapter", max_depth=4)
    if hasattr(pl_model, "brain"):
        tot_b, trn_b = count_params(pl_model.brain)
        print(f"[brain ] total params: {_format_num(tot_b)} | trainable: {_format_num(trn_b)}")
        print_module_tree(pl_model.brain, name="brain", max_depth=6)

    # 只 hook brain 的叶子层就够了（否则输出爆炸）
    hooks = register_shape_hooks(pl_model.brain, focus_keys=None)
    try:
        infer_one_pass(pl_model, sample_batch, device)
    finally:
        for h in hooks:
            h.remove()

    print("================= END REPORT =================\n")

def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        # v 可能是 dict，也可能是 OmegaConf 的 DictConfig
        if isinstance(v, (dict, DictConfig)) and ("target" in v):
            print(f"init {k}")
            model[k] = instantiate_from_config(v)
        else:
            print(f"skip {k} (type={type(v).__name__})")

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model

class EpochSummaryPrinter(pl.Callback):
    def __init__(self, print_val=True):
        super().__init__()
        self.print_val = print_val
        self._w0 = None  # 用来跟踪 brain 参数是否变化

    @staticmethod
    def _get(metrics: dict, *names, default=None):
        """从 callback_metrics 里按可能的名字取值，并转成 float。"""
        for n in names:
            if n in metrics:
                v = metrics[n]
                if isinstance(v, torch.Tensor):
                    v = v.detach().float().cpu().item()
                return v
        return default

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # 记录 brain 第一个参数，用来计算参数更新幅度（是否真的在学）
            p = next(pl_module.brain.parameters())
            self._w0 = p.detach().float().cpu().clone()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics

        # 取常见的 epoch 聚合名字（不同 lightning 版本名字略有差异，所以都兼容）
        train_loss = self._get(metrics, "train_loss_epoch", "train_loss")
        clip_loss  = self._get(metrics, "clip_loss_epoch", "clip_loss")
        txt_loss   = self._get(metrics, "txt_clip_loss_epoch", "txt_clip_loss", default=0.0)
        bt_loss    = self._get(metrics, "bt_loss_epoch", "bt_loss", default=0.0)


        # logit_scale（如果存在）
        try:
            logit_scale = pl_module.brain.softplus(pl_module.brain.logit_scale).detach().float().cpu().item()
        except Exception:
            logit_scale = float("nan")

        # 参数变化量（判断是否真的更新）
        try:
            p = next(pl_module.brain.parameters()).detach().float().cpu()
            delta = (p - self._w0).norm().item() if self._w0 is not None else float("nan")
        except Exception:
            delta = float("nan")

        e = trainer.current_epoch + 1
        E = trainer.max_epochs
        print(
            f"[Train][Epoch {e:03d}/{E:03d}] "
            f"train_loss={train_loss:.6f} | clip={clip_loss:.6f} | txt={txt_loss:.6f} | bt={bt_loss:.6f} | "
            f"logit_scale={logit_scale:.4f} | brain_delta={delta:.6f}"
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if (not self.print_val) or (not trainer.is_global_zero):
            return
        metrics = trainer.callback_metrics
        val_loss = self._get(metrics, "val_loss", "val_loss_epoch", default=float("nan"))
        val_top1 = self._get(metrics, "val_top1_acc", "val_top1_acc_epoch", default=float("nan"))
        val_top5 = self._get(metrics, "val_top5_acc", "val_top5_acc_epoch", default=float("nan"))
        e = trainer.current_epoch + 1
        print(f"[Val  ][Epoch {e:03d}] val_loss={val_loss:.6f} | val_top1={val_top1:.4f} | val_top5={val_top5:.4f}")


class PLModel(pl.LightningModule):
    def __init__(self, model,config,train_loader,test_loader, model_type = 'RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss2()

        self.all_predicted_classes = []
        self.all_true_labels = []
    
        self.z_dim = self.config['z_dim']
        self.c_num = self.config['models']['brain']['params']['c_num']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3
        
        self.mAP_total = 0
        self.match_similarities = []
        # =========================
        # 5) Barlow Twins / Whitening-style decorrelation regularizer
        # =========================
        hp_cfg = self.config.get('model', {})
        if isinstance(hp_cfg, DictConfig):
            hp_cfg = OmegaConf.to_container(hp_cfg, resolve=True)
        elif not isinstance(hp_cfg, dict):
            hp_cfg = {}

        # （可选）如果你还想兼容把超参写到 models 里：
        if len(hp_cfg) == 0:
            m = self.config.get('models', {})
            hp_cfg = OmegaConf.to_container(m, resolve=True) if isinstance(m, DictConfig) else (m if isinstance(m, dict) else {})

        bt_cfg = hp_cfg  # 允许你把参数写进 config['model']
        self.bt_weight = float(bt_cfg.get('bt_weight', 0.05))      # 总权重
        self.bt_lambda = float(bt_cfg.get('bt_lambda', 0.005))     # off-diagonal 权重
        self.bt_eps    = float(bt_cfg.get('bt_eps', 1e-9))

        # 用 projector 把 D=1024 -> 256，相关矩阵更稳也更省
        self.bt_use_projector = bool(bt_cfg.get('bt_use_projector', True))
        self.bt_dim = int(bt_cfg.get('bt_dim', 256))

        # ✅ 关键：只有 bt_weight>0 才创建有参数模块，避免 DDP find_unused_parameters=False 时挂掉
        if self.bt_weight > 0 and self.bt_use_projector:
            self.bt_proj = nn.Sequential(
                nn.LayerNorm(self.z_dim),
                nn.Linear(self.z_dim, self.bt_dim),
                nn.GELU(),
                nn.Linear(self.bt_dim, self.bt_dim),
            )
            self.bt_out_dim = self.bt_dim
        else:
            self.bt_proj = nn.Identity()
            self.bt_out_dim = self.z_dim

        # --- EEG augment 超参（轻量、适合 EEG）---
        aug_cfg = hp_cfg.get('eeg_aug', {})
        self.aug_noise_std = float(aug_cfg.get('noise_std', 0.01))  # 高斯噪声
        self.aug_chan_drop = float(aug_cfg.get('chan_drop', 0.1))   # 通道 dropout 概率
        self.aug_time_mask = float(aug_cfg.get('time_mask', 0.1))   # 时间 mask 比例
        self.aug_max_shift = int(aug_cfg.get('max_shift', 10))      # 时间平移（采样点）
        # =========================
        # 7) CLIP 三模态对齐：EEG↔Text
        # =========================
        txt_cfg = hp_cfg

        # λ_txt：建议从 0.25/0.5/1.0 试
        self.lambda_txt = float(txt_cfg.get('lambda_txt', 0.5))

        # ✅ 强烈建议：文本用 multi-positive（同类多正样本），更符合你数据
        self.txt_multi_positive = bool(txt_cfg.get('txt_multi_positive', True))

        # 数值稳定用
        self.txt_eps = float(txt_cfg.get('txt_eps', 1e-8))

        # =========================
        # 9) Batch Whitening / ZCA (lightweight)
        # =========================
        self.whiten_on = bool(hp_cfg.get('whiten_on', True))          # 总开关
        self.whiten_where = str(hp_cfg.get('whiten_where', 'eeg'))    # 'eeg' 或 'bt'（对 bt_proj 输出 whiten）
        self.whiten_mode = str(hp_cfg.get('whiten_mode', 'bn'))       # 'bn' 或 'zca'
        self.whiten_detach_stats = bool(hp_cfg.get('whiten_detach_stats', False))
        self.whiten_eps = float(hp_cfg.get('whiten_eps', 1e-5))
        self.whiten_shrink = float(hp_cfg.get('whiten_shrink', 0.05)) # cov shrinkage (0~0.2 常用)
        # =========================
        # 10) Orthogonal Procrustes Alignment (learn an orthogonal rotation)
        # =========================
        self.proc_weight = float(hp_cfg.get('proc_weight', 0.02))     # 损失权重（建议小）
        self.proc_dim = int(hp_cfg.get('proc_dim', 256))              # 低维对齐空间
        self.proc_use_ln = bool(hp_cfg.get('proc_use_ln', True))      # projector 是否 LayerNorm
        self.proc_detach_img = bool(hp_cfg.get('proc_detach_img', True))  # img_features 固定，默认 detach 更稳

        # 两个 projector：把 EEG / IMG 都投到同一维度 d=proc_dim
        if self.proc_weight > 0:
            def _proj(in_dim, out_dim):
                if self.proc_use_ln:
                    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
                return nn.Linear(in_dim, out_dim)

            self.proc_proj_eeg = _proj(self.z_dim, self.proc_dim)
            self.proc_proj_img = _proj(self.z_dim, self.proc_dim)

            # 可学习矩阵 A，用 QR 得到正交矩阵 W
            # 用 eye 初始化更稳定（起点接近单位变换）
            A = torch.eye(self.proc_dim)
            self.proc_A = nn.Parameter(A)  # [d,d]
        # =========================
        # 10) Image Adapter / Projector (Route A)
        # =========================
        self.img_adapter_on = bool(hp_cfg.get('img_adapter_on', True))
        self.img_adapter_type = str(hp_cfg.get('img_adapter_type', 'linear'))   # 'linear' or 'mlp'
        self.img_adapter_hidden = int(hp_cfg.get('img_adapter_hidden', self.z_dim))
        self.img_adapter_dropout = float(hp_cfg.get('img_adapter_dropout', 0.1))
        self.img_adapter_residual = bool(hp_cfg.get('img_adapter_residual', True))
        if self.img_adapter_on:
            if self.img_adapter_type == 'linear':
                self.img_adapter = nn.Sequential(
                    # nn.GELU(),
                    nn.Linear(self.z_dim, self.z_dim),
                    # nn.Dropout(0.3),
                    nn.LayerNorm(self.z_dim)
                )
            else:
                self.img_adapter = ResidualMLPAdapter(
                    dim=self.z_dim,
                    hidden_dim=self.img_adapter_hidden,
                    dropout=self.img_adapter_dropout,
                    use_ln=True,
                    residual=self.img_adapter_residual
                )
        else:
            self.img_adapter = nn.Identity()
        self.eeg_adapter_on = bool(hp_cfg.get('eeg_adapter_on', True))
        self.eeg_adapter_type = str(hp_cfg.get('eeg_adapter_type', 'linear'))   # 'linear' or 'mlp'
        self.eeg_adapter_hidden = int(hp_cfg.get('eeg_adapter_hidden', self.z_dim))
        self.eeg_adapter_dropout = float(hp_cfg.get('eeg_adapter_dropout', 0.1))
        self.eeg_adapter_residual = bool(hp_cfg.get('eeg_adapter_residual', True))
        if self.eeg_adapter_on:
            if self.eeg_adapter_type == 'linear':
                self.eeg_adapter = nn.Sequential(
                    # nn.GELU(),
                    nn.Linear(self.z_dim, self.z_dim),
                    # nn.Dropout(0.3),
                    nn.LayerNorm(self.z_dim)
                )
            else:
                self.eeg_adapter = ResidualMLPAdapter(
                    dim=self.z_dim,
                    hidden_dim=self.eeg_adapter_hidden,
                    dropout=self.eeg_adapter_dropout,
                    use_ln=True,
                    residual=self.eeg_adapter_residual
                )
        else:
            self.eeg_adapter = nn.Identity()
        # =========================
        # 11) CCA-style low-rank adapter
        # =========================
        self.cca_adapter_on = bool(hp_cfg.get('cca_adapter_on', True))
        self.cca_rank = int(hp_cfg.get('cca_rank', 128))
        self.cca_dropout = float(hp_cfg.get('cca_dropout', 0.1))
        self.cca_loss_weight = float(hp_cfg.get('cca_loss_weight', 0.05))
        self.cca_lambda_offdiag = float(hp_cfg.get('cca_lambda_offdiag', 0.005))
        if self.cca_adapter_on:
            self.eeg_cca_adapter = CCAStyleAdapter(
                dim=self.z_dim,
                rank=self.cca_rank,
                dropout=self.cca_dropout,
                residual=True,
                use_ln=True
            )
            self.img_cca_adapter = CCAStyleAdapter(
                dim=self.z_dim,
                rank=self.cca_rank,
                dropout=self.cca_dropout,
                residual=True,
                use_ln=True
            )
        else:
            self.eeg_cca_adapter = nn.Identity()
            self.img_cca_adapter = nn.Identity()
        # =========================
        # 12) Amplitude-Phase conditioning branch
        # =========================
        self.amp_phase_on = bool(hp_cfg.get('amp_phase_on', True))
        self.amp_phase_cond_dim = int(hp_cfg.get('amp_phase_cond_dim', 128))
        self.amp_phase_hidden_dim = int(hp_cfg.get('amp_phase_hidden_dim', 64))
        self.amp_phase_film_hidden = int(hp_cfg.get('amp_phase_film_hidden', 128))
        self.amp_phase_reg_weight = float(hp_cfg.get('amp_phase_reg_weight', 0.1))

        self.amp_phase_stats_reduce = str(hp_cfg.get('amp_phase_stats_reduce', 'channel'))

        if self.amp_phase_on:
            self.amp_phase_encoder = AmpPhaseDistributionConditioner(
                n_channels=self.c_num,
                n_times=self.config['data']['timesteps'][1] - self.config['data']['timesteps'][0],
                cond_dim=self.amp_phase_cond_dim,
                hidden_dim=self.amp_phase_hidden_dim,
                stats_reduce=self.amp_phase_stats_reduce,
                use_layernorm=True,
                eps=1e-6
            )
            self.amp_phase_modulator = FiLMModulator(
                z_dim=self.z_dim,
                cond_dim=self.amp_phase_cond_dim,
                hidden_dim=self.amp_phase_film_hidden
            )
        else:
            self.amp_phase_encoder = None
            self.amp_phase_modulator = None
        # =========================
        # 13) Reliability-aware adaptive temperature
        # =========================
        self.adaptive_temp_on = bool(hp_cfg.get('adaptive_temp_on', True))
        self.adaptive_temp_detach_rel = bool(hp_cfg.get('adaptive_temp_detach_rel', True))
        self.adaptive_temp_tau_min = float(hp_cfg.get('adaptive_temp_tau_min', 0.01))
        self.adaptive_temp_init_a = float(hp_cfg.get('adaptive_temp_init_a', -2.0))
        self.adaptive_temp_init_b = float(hp_cfg.get('adaptive_temp_init_b', 1.0))

        if self.adaptive_temp_on:
            self.adaptive_clip_criterion = AdaptiveClipLoss()
            self.adaptive_temp_head = AdaptiveTemperature(
                init_a=self.adaptive_temp_init_a,
                init_b=self.adaptive_temp_init_b,
                tau_min=self.adaptive_temp_tau_min
            )
        else:
            self.adaptive_clip_criterion = None
            self.adaptive_temp_head = None
    def proc_orthogonal_W(self):
        """
        Use QR decomposition to obtain an orthogonal matrix W from learnable A.
        W is [d,d] and approximately orthogonal (QR gives orthonormal Q).
        """
        # torch.linalg.qr supports autograd
        Q, R = torch.linalg.qr(self.proc_A)
        # 处理符号不确定性：让 diag(R) 为正，避免 Q 在训练中跳变
        diag = torch.sign(torch.diagonal(R))
        diag[diag == 0] = 1.0
        Q = Q * diag.unsqueeze(0)
        return Q

    def procrustes_loss(self, eeg_z: torch.Tensor, img_z: torch.Tensor):
        """
        eeg_z/img_z: [B, D] (原始空间)
        1) project -> [B,d]
        2) normalize
        3) rotate eeg by W
        4) MSE in the rotated space
        """
        ze = self.proc_proj_eeg(eeg_z.float())  # [B,d]
        zi = self.proc_proj_img(img_z.float())  # [B,d]

        # if self.proc_detach_img:
        #     zi = zi.detach()

        ze = F.normalize(ze, dim=-1)
        zi = F.normalize(zi, dim=-1)

        W = self.proc_orthogonal_W()            # [d,d]
        ze_rot = ze @ W                         # [B,d]

        # 对齐损失（等价于 2-2*cos，如果都 normalize）
        loss = (ze_rot - zi).pow(2).sum(dim=1).mean()
        return loss
    def batch_whiten(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D]
        mode='bn': per-dim 标准化（BN w/o affine）
        mode='zca': ZCA whitening with shrinkage
        """
        z = z.float()
        B, D = z.shape

        # 1) 去均值
        mu = z.mean(dim=0, keepdim=True)
        x = z - mu

        if self.whiten_detach_stats:
            mu = mu.detach()
            x = (z - mu)

        if self.whiten_mode.lower() == 'bn':
            # per-dim 标准化：等价于 BN(affine=False) 的核心部分
            var = x.var(dim=0, unbiased=False, keepdim=True)
            x = x / torch.sqrt(var + self.whiten_eps)
            return x.type_as(z)

        # 2) ZCA whitening
        # 协方差 [D,D]
        cov = (x.T @ x) / float(B)  # [D,D]
        # shrinkage: cov <- (1-α)cov + αI
        cov = (1.0 - self.whiten_shrink) * cov + self.whiten_shrink * torch.eye(D, device=z.device, dtype=cov.dtype)

        # 3) 计算 cov^{-1/2} via eigh
        # cov = U diag(s) U^T
        cov = cov.double()
        s, U = torch.linalg.eigh(cov)  # s:[D]
        s = torch.clamp(s, min=self.whiten_eps)
        inv_sqrt = (U @ torch.diag(s.rsqrt()) @ U.T).float()  # [D,D]

        # 4) ZCA: x' = x * cov^{-1/2}
        xw = x.float() @ inv_sqrt
        return xw.type_as(z)


    def clip_loss_multi_positive(
        self,
        z_a: torch.Tensor,   # [B, D] e.g. eeg
        z_b: torch.Tensor,   # [B, D] e.g. text
        logit_scale: torch.Tensor,
        labels: torch.Tensor # [B] class id
    ) -> torch.Tensor:
        """
        Multi-positive InfoNCE（同类多正样本）：
        对每个 i，把所有 label 相同的 j 都当正样本，最大化 sum_{j in Pos(i)} p(j|i)

        返回：标量 loss
        """
        # 归一化（更接近 CLIP 的做法，也更稳）
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = logit_scale * (z_a @ z_b.T)  # [B, B]

        labels = labels.view(-1, 1)  # [B,1]
        pos_mask = labels.eq(labels.T).float()  # [B,B] 同类为 1

        # a -> b
        logp_ab = F.log_softmax(logits, dim=1)  # [B,B]
        pos_prob_ab = (pos_mask * logp_ab.exp()).sum(dim=1).clamp_min(self.txt_eps)  # [B]
        loss_ab = (-pos_prob_ab.log()).mean()

        # b -> a（对称）
        logp_ba = F.log_softmax(logits.T, dim=1)
        pos_prob_ba = (pos_mask * logp_ba.exp()).sum(dim=1).clamp_min(self.txt_eps)
        loss_ba = (-pos_prob_ba.log()).mean()

        return 0.5 * (loss_ab + loss_ba)

        
    # ===== Barlow Twins helpers =====
    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_twins_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1,z2: [B, D]
        L = sum_i (1 - C_ii)^2 + λ sum_{i!=j} C_ij^2
        """
        z1 = z1.float()
        z2 = z2.float()

        # per-dim 标准化（等价于 Barlow 里的 BN(affine=False)）
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0, unbiased=False) + self.bt_eps)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0, unbiased=False) + self.bt_eps)

        B = z1.shape[0]
        c = (z1.T @ z2) / float(B)  # [D,D]

        on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        return on_diag + self.bt_lambda * off_diag

    # ===== EEG augment =====
    def augment_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, C, T]
        轻量增强：噪声 + 通道drop + 时间mask + 时间shift
        """
        x = eeg

        # 1) Gaussian noise
        if self.aug_noise_std > 0:
            x = x + self.aug_noise_std * torch.randn_like(x)

        # 2) Channel dropout
        if self.aug_chan_drop > 0:
            B, C, T = x.shape
            keep = (torch.rand(B, C, 1, device=x.device) > self.aug_chan_drop).float()
            x = x * keep

        # 3) Time mask
        if self.aug_time_mask > 0:
            B, C, T = x.shape
            mlen = max(1, int(T * self.aug_time_mask))
            start = torch.randint(low=0, high=max(1, T - mlen), size=(1,), device=x.device).item()
            x = x.clone()
            x[:, :, start:start + mlen] = 0.0

        # 4) Time shift（整 batch 同一个 shift，便宜且有效）
        if self.aug_max_shift > 0:
            shift = int(torch.randint(-self.aug_max_shift, self.aug_max_shift + 1, (1,), device=x.device).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        return x

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.brain(eeg)

    def forward(self, batch,sample_posterior=False):
 
        idx = batch['idx'].cpu().detach().numpy() 
        eeg = batch['eeg']
        img = batch['img']

        img_z  = batch['img_features']
        eeg_z = self.brain(eeg)
        eeg_base_z = eeg_z
        # image adapter: 让固定图像特征可以轻量适配
        img_z = self.img_adapter(img_z)
        eeg_z = self.eeg_adapter(eeg_z)
        eeg_center_z = eeg_z
        eeg_lowrank = None
        img_lowrank = None

        # =========================
        # ✅ Batch Whitening (apply to eeg_z before CLIP)
        # =========================
        if self.whiten_on and self.whiten_where == 'eeg':
            eeg_z = self.batch_whiten(eeg_z)
                
        eeg_z = 0.8 * eeg_z + 0.2 * eeg_center_z + 0.05 * eeg_base_z

        # ---------- CCA-style low-rank adapters ----------
        if self.cca_adapter_on:
            eeg_z, eeg_lowrank = self.eeg_cca_adapter(eeg_z, return_lowrank=True)
            img_z, img_lowrank = self.img_cca_adapter(img_z, return_lowrank=True)
        eeg_cca_z = eeg_z
        # ---------- amplitude-phase conditioning ----------
        amp_phase_cond = None
        amp_phase_stats = None
        film_gamma = None
        film_beta = None
        film_conf = None
        if self.amp_phase_on:
            amp_phase_cond, amp_phase_stats = self.amp_phase_encoder(eeg)
            eeg_z, film_gamma, film_beta, film_conf = self.amp_phase_modulator(eeg_z, amp_phase_cond)
        sample_reliability = None
        if amp_phase_stats is not None:
            sample_reliability = compute_sample_reliability_from_phase_stats(
                amp_phase_stats,
                detach=self.adaptive_temp_detach_rel
            )
        img_z = img_z/img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]
            
            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim -z_alpha_2 * std_sim
            upper_bound = mean_sim +z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label
           
            loss = total_loss
        else:
            loss = total_loss
        return (
            eeg_z, img_z, loss,
            eeg_lowrank, img_lowrank,
            amp_phase_cond, amp_phase_stats,
            film_gamma, film_beta, film_conf,
            sample_reliability
        )
        
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            w = next(self.brain.parameters())
            print("brain weight mean(after epoch) =", w.data.float().mean().item())

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        (
            eeg_z, img_z, clip_loss,
            eeg_lowrank, img_lowrank,
            amp_phase_cond, amp_phase_stats,
            film_gamma, film_beta, film_conf,
            sample_reliability
        ) = self(batch, sample_posterior=True)
        if self.adaptive_temp_on and (sample_reliability is not None):
            tau = self.adaptive_temp_head(sample_reliability)   # [B]

            main_eeg_loss, main_img_loss, _ = self.adaptive_clip_criterion(
                eeg_z,
                img_z,
                tau
            )
            clip_loss = 0.5 * (main_eeg_loss.mean() + main_img_loss.mean())
            loss = clip_loss

            self.log('tau_mean', tau.mean(), on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('tau_min', tau.min(), on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
            self.log('tau_max', tau.max(), on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        else:
            loss = clip_loss

        if self.cca_adapter_on and self.cca_loss_weight > 0 and (eeg_lowrank is not None) and (img_lowrank is not None):
            cca_loss = cca_correlation_loss(
                eeg_lowrank,
                img_lowrank,
                eps=self.txt_eps,
                lambda_offdiag=self.cca_lambda_offdiag
            )
            loss = loss + self.cca_loss_weight * cca_loss
            self.log('cca_loss', cca_loss, on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        if self.amp_phase_on and self.amp_phase_reg_weight > 0 and (film_gamma is not None) and (film_beta is not None):
            amp_phase_reg = ((film_gamma - 1.0).pow(2).mean() + film_beta.pow(2).mean())
            loss = loss + self.amp_phase_reg_weight * amp_phase_reg
            self.log('amp_phase_reg', amp_phase_reg, on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        # if self.amp_phase_on and (amp_phase_stats is not None) and (film_conf is not None):
        #     phase_var = amp_phase_stats["phase_circ_var"].mean(dim=-1, keepdim=True)   # [B,1]
        #     conf_target = 1.0 - phase_var.detach()
        #     conf_reg = (film_conf - conf_target).pow(2).mean()
        #     loss = loss + 0.01 * conf_reg

        #     self.log('amp_phase_conf_reg', conf_reg, on_step=True, on_epoch=True,
        #              prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('clip_loss', clip_loss, on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True,
                prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
    
        outputs = self(batch)
        eeg_z, img_z, loss_img = outputs[0], outputs[1], outputs[2]
        loss = loss_img

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self,batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        outputs = self(batch)
        eeg_z, img_z, loss = outputs[0], outputs[1], outputs[2]
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        # label =  batch['label']
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())


        #compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())


        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        
        return loss
        
    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item(),'mAP':self.mAP,'similarity':self.match_similarities}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'], weight_decay=1e-4)

        return [optimizer]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed,  workers=True)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'],exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}_{run_id}")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)

    print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)
    # ====== 打印模型结构 & 维度 ======
    if True:  # 你想关掉就改 False
        sample_batch = next(iter(train_loader))
        print_model_report(pl_model, sample_batch)

    # ====== 替换你原来的 checkpoint_callback 定义 ======
    checkpoint_callback = ModelCheckpoint(
        monitor='val_top1_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        filename='{epoch}-{step}-{val_top1_acc:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_top1_acc',
        min_delta=0.001,
        patience=30,
        verbose=True,
        mode='max'
    )

    printer_cb = EpochSummaryPrinter(print_val=True)
    trainer = Trainer(
        log_every_n_steps=1,
        strategy=DDPStrategy(process_group_backend="nccl", find_unused_parameters=True),
        callbacks=[early_stop_callback, checkpoint_callback, printer_cb],
        max_epochs=config['train']['epoch'],
        devices=1,
        accelerator='gpu',
        logger=logger,
        enable_progress_bar=False,
         num_sanity_val_steps=0
    )
    print(trainer.logger.log_dir)

    ckpt_path = None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    # ====== 关键：fit 完以后，用单卡重新评估 ======
    best_ckpt = checkpoint_callback.best_model_path
    last_ckpt = checkpoint_callback.last_model_path

    ckpt_to_eval = best_ckpt if (best_ckpt is not None and best_ckpt != "") else last_ckpt
    print(f"[Eval] use checkpoint: {ckpt_to_eval}")

    trainer_eval = Trainer(
        accelerator="gpu",
        devices=1,               # ✅ 单卡评估
        logger=logger,
        enable_checkpointing=False
    )

    # （可选）单卡跑一次 validate，得到真实 val_top1/val_top5
    val_results = trainer_eval.validate(pl_model, dataloaders=val_loader, ckpt_path=ckpt_to_eval)
    print("val_results(single-gpu) =", val_results)

    # 单卡 test（真实 200-way）
    test_results = trainer_eval.test(pl_model, dataloaders=test_loader, ckpt_path=ckpt_to_eval)

    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__=="__main__":
    main()