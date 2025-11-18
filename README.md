# note_for_DA2
My notes for Depth-Anything-V2 for Metric Depth Estimation.

- [Reference](#reference)
- [Introduction](#introduction)
- [System setup](#system-setup)
- [Script](#script)

## Reference
- [Depth Anything V2 for Metric Depth Estimation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)

## Introduction
- Fine-tune **Depth Anything V2** pre-trained encoder on synthetic Hypersim / Virtual KITTI datasets for indoor / outdoor metric depth estimation.
- Use the **DPT** head to regress the depth.
![Compare DA2 with Zoedepth](./images/compare_zoedepth_DA2.png)

## System setup
### Package
```commandline
pip install -r requirements.txt
```

### Weights
Download the checkpoints listed [**here**](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) and put them under the `checkpoints` directory.

## Script
### run.py