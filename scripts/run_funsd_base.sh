#!/bin/bash

set -e

# Đổi đường dẫn làm việc sang đúng thư mục trên Kaggle
cd /kaggle/working/KIE_Layoutlm

# Khai báo biến trỏ thẳng vào Python của Conda
PYTHON_CMD="/kaggle/working/miniconda/envs/layoutlmv3/bin/python"

export PYTHONPATH="/kaggle/working/KIE_Layoutlm:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="FUNSD-Base-Experiment"

SEEDS=(42 123 1993)

for SEED in "${SEEDS[@]}"
do
    OUT="./funsd-base-seed${SEED}"

    echo ""
    echo "============================================================"
    echo "FUNSD BASE - SEED = ${SEED}"
    echo "============================================================"

    if [ -f "$OUT/eval_results.json" ]; then
        echo "[SKIP] Seed ${SEED} đã có kết quả."
        continue
    fi

    rm -rf "$OUT"

    # SỬA LỖI 3: Dùng torchrun thay cho torch.distributed.launch
    $PYTHON_CMD -m torch.distributed.run --nproc_per_node=2 examples/run_funsd_cord.py \
      --dataset_name funsd \
      --do_train \
      --do_eval \
      --do_predict \
      --use_segment_head \
      --model_name_or_path /kaggle/working/layoutlmv3-base-local \
      --output_dir "$OUT" \
      --input_size 224 \
      --max_steps 1000 \
      --save_steps 1000 \
      --evaluation_strategy steps \
      --eval_steps 100 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --dataloader_num_workers 4 \
      --remove_unused_columns False \
      --report_to none \
      --run_name "FUNSD-LR-Split-seed${SEED}" \
      --seed "$SEED" \
      --overwrite_output_dir \
      --overwrite_cache
      # SỬA LỖI 2: Đã loại bỏ --segment_level_layout 1 và --visual_embed 1 (Mặc định đã là True)
done

echo ""
echo "============================================================"
echo "CALCULATING FUNSD BASE 3-SEED MEAN ± STD"
echo "============================================================"

# CŨNG DÙNG CONDA PYTHON CHO ĐOẠN SCRIPT TÍNH TOÁN NÀY
$PYTHON_CMD - <<'PY'
import os
import json
import numpy as np

seeds = [42, 123, 1993]
metrics = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_loss"]
results = {m: [] for m in metrics}

for seed in seeds:
    path = "./funsd-base-seed{}/eval_results.json".format(seed)
    print("\nSeed {}:".format(seed))

    if not os.path.exists(path):
        print("  [WARNING] Missing:", path)
        continue

    with open(path, "r") as f:
        data = json.load(f)

    for metric in metrics:
        if metric in data:
            value = float(data[metric])
            results[metric].append(value)
            print("  {:18s} = {:.6f}".format(metric, value))

print("\n" + "=" * 70)
print("FINAL FUNSD BASE RESULT: MEAN ± STD")
print("=" * 70)

summary = {}
for metric in metrics:
    values = results[metric]
    if not values:
        print("{:18s}: NO DATA".format(metric))
        continue

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    print("{:18s}: {:.4f} ± {:.4f}".format(metric, mean, std))

    summary[metric] = {"values": values, "mean": float(mean), "std": float(std)}

output_summary_file = "funsd_base_3seed_summary.json"
with open(output_summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("Saved:", output_summary_file)
PY