import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

def run_depth_anything(
        img_path = "",
        input_size = 518,
        load_from = "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
        outdir = "./vis_depth",
        encoder = "vitl",
        max_depth = 20,
        save_numpy = False,
        grayscale = False,
        pred_only = False
):
    """
    Run Depth Anything V2 model to estimate depth from images.

    :param img_path: str, Path to a single image file.
    :param input_size: int, Input resolution for Depth Anything V2.
    :param load_from: str, Path to checkpoint (.pth) of Depth Anything V2 weights.
    :param outdir: str, Path to output directory.
    :param encoder: str, Encoder type for Depth Anything V2. Options: "vits", "vitb", "vitl", "vitg"
    :param max_depth: float, Maximum depth value (in meters) used to normalize the predicted depth map.
    :param save_numpy: bool, If True, saves the raw depth (in meters) as .npy files.
    :param grayscale: bool, If True, outputs grayscale depth maps instead of colorful colormap.
    :param pred_only:bool, If True, shows only the depth visualization.
    :return: None
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

    os.makedirs(outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, input_size)

        if save_numpy:
            output_path = os.path.join(outdir,
                                       os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)

if __name__ == '__main__':
    run_depth_anything(
        img_path=f"test_img/M1_11_intensity_image.png",
        input_size=518,
        load_from="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
        outdir="./vis_depth_2meter",
        encoder="vitl",
        max_depth=5,
        save_numpy=True,
        grayscale=False,
        pred_only=False
    )