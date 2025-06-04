以下是将你提供的内容重新改写为更贴合**SCI风格项目在 GitHub 上发布**的 `README.md` 格式，语言更加专业、结构更清晰，也更便于读者快速理解和复现：

------

# VLBCD

A Vision-Language Bimodal Change Detection Framework based on Box-Level Supervision.

## 1. Environment Setup

**Supported Platforms:**

- Linux / macOS
- Python ≥ 3.6 (Tested on Python 3.8)
- PyTorch ≥ 1.7 (Tested on PyTorch 1.9.0)
- CUDA ≥ 9.2 (Tested on CUDA 11.1)
- GCC ≥ 5.0

**Dependencies:**

- `mmdet==2.25.0`
- `mmcv-full==1.5.0`
- `mmseg==0.20.0`
- `regex`, `ftfy`, `fvcore`
- `timm==0.6.13`

### Installation Guide

This project is developed based on [BoxInstSeg](https://github.com/LiWentomng/BoxInstSeg). Follow the steps below to set up the environment:

#### Step 1: Create Conda Environment

```bash
conda create -n vlbcd python=3.8 -y
conda activate vlbcd
```

#### Step 2: Install PyTorch, torchvision and torchaudio

Refer to the [official installation guide](https://pytorch.org/get-started/previous-versions/). For CUDA 11.1:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Step 3: Install MMCV via OpenMIM

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
```

#### Step 4: Clone the Repository

```bash
git clone https://github.com/TiffenYi/VLBCD
cd VLBCD
```

#### Step 5: Install Dependencies and MMDetection

```bash
bash setup.sh
```

------

## 2. Dataset Structure

The dataset should follow the structure below:

```
dataset/
├── annotations/       # Box-level COCO-style .json annotations
├── train/
│   ├── A/             # First temporal image
│   ├── B/             # Second temporal image
│   └── label/         # Pixel-wise labels (optional, for comparison)
└── val/
    ├── A/
    ├── B/
    └── label/
```

> `annotations/` contains COCO-style `.json` files used for training with box-level supervision.
>  `label/` folders contain optional pixel-level labels for evaluation or comparison.

------

## 3. Usage

### 3.1 Training

To train the model using a specified config file:

```bash
python tools/train.py path/to/your_config.py
```

### 3.2 Inference and Visualization

To run inference using a trained model and save the prediction visualizations:

```bash
python box2cd_result.py path/to/config.py path/to/model.pth --show-dir path/to/output_images/
```

- `path/to/config.py`: configuration file used for training
- `path/to/model.pth`: checkpoint file
- `--show-dir`: directory to save the visualization outputs



## 4. Acknowledgements

This work is based on [BoxInstSeg](https://github.com/LiWentomng/BoxInstSeg). We thank the authors for their open-source contribution.



