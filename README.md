# EUEA: Environmental Understanding Vision-Language Model for Embodied Agent

Official repository of the **EUEA** paper accepted at **CVPRF 2026**.

---

## Overview

EUEA is a Vision-Language Model (VLM) designed for environmental understanding in embodied AI settings. It integrates a 3D-aware visual encoder (ViT + PointNet) with a large language model backbone (LLaMA-3) via a learned projection layer, enabling spatial question answering and scene captioning directly from RGB images and point clouds.

Supported tasks and datasets:
| Dataset | Task |
|---|---|
| [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) | 3D QA, Visual Grounding, Dense Captioning |
| [ScanQA](https://github.com/ATR-DBI/ScanQA) | 3D Question Answering |
| [SQA3D](https://github.com/SilongYong/SQA3D) | Situated 3D Question Answering |

---

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
```

---

## Dataset Download

Download and prepare all datasets with:

```bash
python scripts/download_datasets.py --data_root /data/euea
```

Download a specific dataset:

```bash
python scripts/download_datasets.py --data_root /data/euea --datasets scanqa sqa3d
```

Print information and instructions without downloading:

```bash
python scripts/download_datasets.py --data_root /data/euea --info_only
```

> **Note**: ScanQA and SQA3D require the raw ScanNet scans. After obtaining access at [http://www.scan-net.org](http://www.scan-net.org), preprocess the scans into the expected format:
> ```bash
> python scripts/preprocess_scannet.py \
>     --scannet_root /data/scannet \
>     --output_root /data/euea \
>     --num_points 40000
> ```

Expected data layout after setup:

```
/data/euea/
├── embodiedscan/
│   ├── embodiedscan_infos_train.json
│   ├── embodiedscan_infos_val.json
│   ├── embodiedscan_infos_test.json
│   ├── images/<scene_id>/<frame_id>.jpg
│   └── pointclouds/<scene_id>.npy
├── scanqa/
│   ├── ScanQA_v1.0_train.json
│   ├── ScanQA_v1.0_val.json
│   ├── ScanQA_v1.0_test_w_obj.json
│   └── scans/<scene_id>/
│       ├── color/<frame_id>.jpg
│       └── <scene_id>_pointcloud.npy
└── sqa3d/
    ├── sqa_task/
    │   ├── questions/
    │   └── balanced/
    └── scans/<scene_id>/
        ├── color/<frame_id>.jpg
        └── <scene_id>_pointcloud.npy
```

---

## Training

Edit `configs/train_config.yaml` to set your data root, dataset, and hyperparameters, then run:

```bash
# Single GPU
python train.py --config configs/train_config.yaml

# Multi-GPU with accelerate
accelerate launch train.py --config configs/train_config.yaml

# Override config values inline
python train.py --config configs/train_config.yaml \
    data.dataset=scanqa training.num_epochs=10 training.batch_size=16
```

Key training configuration options (`configs/train_config.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `data.dataset` | `embodiedscan` | Dataset to train on (`embodiedscan` / `scanqa` / `sqa3d`) |
| `data.data_root` | `/data/euea` | Path to the dataset root directory |
| `training.num_epochs` | `10` | Number of training epochs |
| `training.batch_size` | `8` | Per-GPU batch size |
| `training.learning_rate` | `1e-4` | Peak learning rate |
| `model.language_decoder.model_name` | `meta-llama/Llama-3.1-8B-Instruct` | LLM backbone |
| `model.language_decoder.use_lora` | `true` | Use LoRA for parameter-efficient fine-tuning |

---

## Evaluation

```bash
# Evaluate on validation split
python eval.py --config configs/eval_config.yaml

# Evaluate on test split
python eval.py --config configs/eval_config.yaml evaluation.split=test

# Save visualizations alongside predictions
python eval.py --config configs/eval_config.yaml evaluation.visualize=true
```

Computed metrics: Exact Match (EM), CIDEr, BLEU-1/2/3/4, METEOR.

Results are saved to `outputs/euea_eval/metrics.json`.

---

## Repository Structure

```
EUEA/
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── requirements.txt
├── configs/
│   ├── train_config.yaml
│   └── eval_config.yaml
├── models/
│   ├── euea.py               # Top-level EUEA model
│   ├── visual_encoder.py     # ViT + PointNet visual encoder
│   └── language_decoder.py   # LLM decoder with LoRA support
├── datasets/
│   ├── base_dataset.py       # Base dataset class
│   ├── embodied_scan.py      # EmbodiedScan dataset
│   ├── scanqa.py             # ScanQA dataset
│   └── sqa3d.py              # SQA3D dataset
├── utils/
│   ├── metrics.py            # EM, CIDEr, BLEU, METEOR
│   └── visualization.py      # Point cloud and prediction visualization
└── scripts/
    ├── download_datasets.py  # Dataset download helper
    └── preprocess_scannet.py # ScanNet scan preprocessor
```

---

## Citation

If you use EUEA in your research, please cite:

```bibtex
@inproceedings{euea2026,
  title     = {Environmental Understanding Vision-Language Model for Embodied Agent},
  booktitle = {CVPRF},
  year      = {2026},
}
```
