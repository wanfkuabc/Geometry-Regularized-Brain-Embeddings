#!/usr/bin/env bash
set -euo pipefail

# =========================
# Default config
# =========================
CONFIG="configs/meg/ubp_zca_qr_img_eegprj.yaml"
PY="python"
SCRIPT="main_barlow_text_zca_qr_img_eegprj_cca_attn_dis_t.py"

# BRAIN_BACKBONE="EEGProjectLayer"
BRAIN_BACKBONE="BaseModel"
# BRAIN_BACKBONE="BrainMLP"

EXP_SETTING="intra-subject"
VISION_BACKBONE="RN50"
# VISION_BACKBONE="ViT-H-14"
EPOCH=40
LR="5e-5"
SEED_LIST=("0")
CUDA_DEV="0"

# default subjects 01..10
SUBJECT_LIST=("01" "02" "03" "04")

LOG_ROOT="logs"

usage() {
  cat <<EOF
Usage:
  bash train.sh [options]

Options:
  --sub 08                 Run single subject (e.g., 08)
  --subs 01,02,08          Run multiple subjects (comma-separated)
  --all                    Run all subjects (default: 01..10)
  --seeds 0,1,2            Run multiple seeds (comma-separated)
  --cuda 0                 CUDA_VISIBLE_DEVICES (default: 0)
  --config path.yaml       Config path (default: configs/eeg/ubp.yaml)
  --script file.py         Train script (default: main_fbcsp_barlow_best.py)
  --brain EEGProjectLayer  Brain backbone
  --vision RN50            Vision backbone
  --exp intra-subject      exp_setting
  --epoch 50               epochs
  --lr 1e-4                learning rate
  --logdir logs            log root
  -h, --help               Show help

Examples:
  # run all subjects, seed=0 on cuda 0
  bash train.sh --all --cuda 0

  # run single subject 08
  bash train.sh --sub 08 --cuda 0

  # run subjects 01,02,08 with seeds 0 and 1
  bash train.sh --subs 01,02,08 --seeds 0,1 --cuda 0
EOF
}

# =========================
# Parse args
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sub)
      SUBJECT_LIST=("$2")
      shift 2
      ;;
    --subs)
      IFS=',' read -r -a SUBJECT_LIST <<< "$2"
      shift 2
      ;;
    --all)
      SUBJECT_LIST=("01" "02" "03" "04")
      shift 1
      ;;
    --seeds)
      IFS=',' read -r -a SEED_LIST <<< "$2"
      shift 2
      ;;
    --cuda)
      CUDA_DEV="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --script)
      SCRIPT="$2"
      shift 2
      ;;
    --brain)
      BRAIN_BACKBONE="$2"
      shift 2
      ;;
    --vision)
      VISION_BACKBONE="$2"
      shift 2
      ;;
    --exp)
      EXP_SETTING="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --logdir)
      LOG_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_ROOT"

echo "======== RUN CONFIG ========"
echo "SCRIPT         = $SCRIPT"
echo "CONFIG         = $CONFIG"
echo "CUDA           = $CUDA_DEV"
echo "BRAIN_BACKBONE = $BRAIN_BACKBONE"
echo "VISION_BACKBONE= $VISION_BACKBONE"
echo "EXP_SETTING    = $EXP_SETTING"
echo "EPOCH          = $EPOCH"
echo "LR             = $LR"
echo "SUBJECTS       = ${SUBJECT_LIST[*]}"
echo "SEEDS          = ${SEED_LIST[*]}"
echo "LOG_ROOT       = $LOG_ROOT"
echo "============================"

# =========================
# Run loop
# =========================
for seed in "${SEED_LIST[@]}"; do
  for sub in "${SUBJECT_LIST[@]}"; do
    sub2=$(printf "%02d" "$((10#$sub))")  # normalize like 08 -> 08

    run_name="sub-${sub2}_seed${seed}_${BRAIN_BACKBONE}_${VISION_BACKBONE}"
    log_file="${LOG_ROOT}/${run_name}.log"

    echo ""
    echo ">>> [RUN] ${run_name}"
    echo ">>> log: ${log_file}"

    # If you want to keep going even if one run fails, don't use 'set -e' around python call
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PY} "${SCRIPT}" \
      --dataset meg \
      --config "${CONFIG}" \
      --subjects "sub-${sub2}" \
      --seed "${seed}" \
      --exp_setting "${EXP_SETTING}" \
      --brain_backbone "${BRAIN_BACKBONE}" \
      --vision_backbone "${VISION_BACKBONE}" \
      --epoch "${EPOCH}" \
      --lr "${LR}" \
      2>&1 | tee "${log_file}"
    ret=${PIPESTATUS[0]}
    set -e

    if [[ $ret -ne 0 ]]; then
      echo "!!! [FAIL] ${run_name} (exit=${ret})"
      # continue to next subject/seed
    else
      echo "+++ [OK] ${run_name}"
    fi
  done
done

echo ""
echo "All jobs done."
