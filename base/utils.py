import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import importlib
import cv2
from PIL import Image
import subprocess
import math

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if config["target"] == "base._pipeline.StableDiffusionXLPipeline":
        return get_obj_from_str(config["target"]).from_pretrained(**config.get("params", dict()) if config.get("params", dict()) else {})
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()) if config.get("params", dict()) else {})

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key):
            if getattr(args, key) != None:
                config[key] = getattr(args, key)
    for key in args.__dict__.keys():
        config[key]=getattr(args, key)
    return config


def get_device(gpu_ids):
    if gpu_ids=='auto':
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'])
        gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
        gpu_info = []
        for line in gpu_info_lines:
            gpu_data = line.strip().split(', ')
            index, memory_free, temperature = map(int, gpu_data)
            gpu_info.append((index, memory_free, temperature))
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        
        memeory_rank_num=math.ceil(0.4*len(gpu_info))
        selected_gpus = gpu_info[:memeory_rank_num]
        selected_gpus.sort(key=lambda x: x[2])
        selected_device = selected_gpus[0][0]
        # device = torch.device(f'cuda:{selected_device}')
    elif gpu_ids=="cpu":
        device = torch.device('cpu')
    else:
        gpu_ids = list(map(int,gpu_ids.split(",")))
        selected_device=gpu_ids[0]
        # device = torch.device(f'cuda:{selected_device}')
    return selected_device

class SigLipLoss(nn.Module):
    def __init__(self, normalize: bool = True, eps: float = 1e-8, bias_init: float = 0.0):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        # 可学习 bias，SigLIP 类做法里常见
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, image_features, text_features, logit_scale):
        """
        image_features: [B, D]
        text_features : [B, D]
        logit_scale   : scalar, usually positive
        """
        device = image_features.device
        B = image_features.shape[0]

        if self.normalize:
            image_features = F.normalize(image_features, dim=-1, eps=self.eps)
            text_features  = F.normalize(text_features,  dim=-1, eps=self.eps)

        logits = logit_scale * (image_features @ text_features.T) + self.bias  # [B, B]

        # 对角线是正样本，其余是负样本
        labels = torch.eye(B, device=device)

        # pairwise sigmoid / BCE with logits
        loss_mat = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        # 可分别统计正负样本损失
        pos_mask = labels
        neg_mask = 1.0 - labels

        pos_loss = (loss_mat * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
        neg_loss = (loss_mat * neg_mask).sum() / neg_mask.sum().clamp_min(1.0)

        total_loss = pos_loss + neg_loss
        return total_loss, pos_loss, neg_loss, logits
class SigLipMultiPositiveLoss(nn.Module):
    def __init__(self, normalize: bool = True, eps: float = 1e-8, bias_init: float = 0.0):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, feat_a, feat_b, logit_scale, labels):
        device = feat_a.device

        if self.normalize:
            feat_a = F.normalize(feat_a, dim=-1, eps=self.eps)
            feat_b = F.normalize(feat_b, dim=-1, eps=self.eps)

        logits = logit_scale * (feat_a @ feat_b.T) + self.bias  # [B,B]

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float().to(device)
        neg_mask = 1.0 - pos_mask

        loss_mat = F.binary_cross_entropy_with_logits(logits, pos_mask, reduction='none')

        pos_loss = (loss_mat * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
        neg_loss = (loss_mat * neg_mask).sum() / neg_mask.sum().clamp_min(1.0)

        total_loss = pos_loss + neg_loss
        return total_loss, pos_loss, neg_loss, logits

class ClipLoss(nn.Module):
    def __init__(self, normalize: bool = False, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
       
    def compute_ranking_weights(self,loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1 / (i + 1)
        return weights
    
    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        # ✅ 在 loss 内显式 L2 归一化（CLIP 标准做法）
        if self.normalize:
            image_features = F.normalize(image_features, dim=-1, eps=self.eps)
            text_features  = F.normalize(text_features,  dim=-1, eps=self.eps)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')

        # total_loss = (image_loss + text_loss) / 2
        
        return image_loss,text_loss, logits_per_image
    
class ClipLoss2(nn.Module):
    def __init__(self, normalize: bool = False, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device

        if self.normalize:
            image_features = F.normalize(image_features, dim=-1, eps=self.eps)
            text_features  = F.normalize(text_features,  dim=-1, eps=self.eps)

        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text  = logits_per_image.T

        labels = torch.arange(logits_per_image.size(0), device=device)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss  = F.cross_entropy(logits_per_text, labels, reduction='none')

        return image_loss, text_loss, logits_per_image

class ClipLoss3(nn.Module):
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def compute_ranking_weights(self, loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1.0 / (i + 1)
        return weights

    def forward(self, image_features, text_features, logit_scale):
        """
        image_features: [N, D]
        text_features : [N, D]
        logit_scale   : scalar, usually exp(logit_scale_param) or softplus(...)

        Returns:
            image_loss: [N]
            text_loss : [N]
            logits_per_image: [N, N]
        """
        if self.normalize:
            image_features = F.normalize(image_features, dim=-1, eps=self.eps)
            text_features  = F.normalize(text_features,  dim=-1, eps=self.eps)

        # pairwise similarity / temperature scaling
        logits_per_image = logit_scale * (image_features @ text_features.T)   # [N, N]
        logits_per_text  = logits_per_image.T                                 # [N, N]

        N = logits_per_image.size(0)
        device = logits_per_image.device

        # positive logits: diagonal
        pos_i2t = logits_per_image.diag()   # [N]
        pos_t2i = logits_per_text.diag()    # [N]

        # mask out diagonal -> keep negatives only
        neg_mask = ~torch.eye(N, dtype=torch.bool, device=device)  # [N, N]

        neg_i2t = logits_per_image.masked_select(neg_mask).view(N, N - 1)  # [N, N-1]
        neg_t2i = logits_per_text.masked_select(neg_mask).view(N, N - 1)   # [N, N-1]

        # denominator = sum over negatives only
        denom_i2t = torch.exp(neg_i2t).sum(dim=1).clamp_min(self.eps)  # [N]
        denom_t2i = torch.exp(neg_t2i).sum(dim=1).clamp_min(self.eps)  # [N]

        # numerator = positive only
        num_i2t = torch.exp(pos_i2t)
        num_t2i = torch.exp(pos_t2i)

        # loss exactly following your formula
        image_loss = -torch.log(num_i2t / denom_i2t + self.eps)   # [N]
        text_loss  = -torch.log(num_t2i / denom_t2i + self.eps)   # [N]

        return image_loss, text_loss, logits_per_image
    
class ClipLoss4(nn.Module):
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def compute_ranking_weights(self, loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1.0 / (i + 1)
        return weights

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device

        if self.normalize:
            image_features = F.normalize(image_features, dim=-1, eps=self.eps)
            text_features  = F.normalize(text_features, dim=-1, eps=self.eps)

        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text  = logits_per_image.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss  = F.cross_entropy(logits_per_text, labels, reduction='none')

        return image_loss, text_loss, logits_per_image