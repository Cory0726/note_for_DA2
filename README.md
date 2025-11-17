# note_for_DA2
My notes for Depth-Anything-V2 for Metric Depth Estimation.

## Reference
- [Depth Anything V2 for Metric Depth Estimation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)

## Introduction
- Fine-tune **Depth Anything V2** pre-trained encoder on synthetic Hypersim / Virtual KITTI datasets for indoor / outdoor metric depth estimation.
- Use the **DPT** head to regress the depth.
![Compare DA2 with Zoedepth](./images/compare_zoedepth_DA2.png)

## System setup
### Package
```commandline
# Pytorch
pip install "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" --index-url https://download.pytorch.org/whl/cu128
# xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps
# others
pip install matplotlib
pip install opencv-python
```