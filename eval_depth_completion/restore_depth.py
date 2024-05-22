#!/usr/bin/env python3
'''Script to run Depth Completion on Synthetic and Real datasets,modified input and output depth images.
'''
import argparse
import csv
import glob
import os
import shutil
import sys
import itertools

# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import depth_completion_api
from api import utils as api_utils

import attrdict
import imageio
import termcolor
import yaml
import torch
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
parser.add_argument('-m', '--maskInputDepth', action="store_true", help='Whether we should mask out objects in input depth')
args = parser.parse_args()

# Load Config File
CONFIG_FILE_PATH = args.configFile      ## config.yaml 불어오는 파일
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = attrdict.AttrDict(config_yaml)

# Create directory to save results
RESULTS_ROOT_DIR = config.resultsDir
runs = sorted(glob.glob(os.path.join(RESULTS_ROOT_DIR, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id))
if os.path.isdir(results_dir):
    if len(os.listdir(results_dir)) > 1:
        # Min 1 file always in folder: copy of config file
        results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id + 1))
        os.makedirs(results_dir)
else:
    os.makedirs(results_dir)
shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('\nSaving results to folder: ' + termcolor.colored('"{}"\n'.format(results_dir), 'green'))

# Init depth completion API
outputImgHeight = int(config.depth2depth.yres)
outputImgWidth = int(config.depth2depth.xres)

# Create lists of input data
rgb_file_list = []
depth_file_list = []
segmentation_masks_list = []
gt_depth_file_list = []
for dataset in config.files:
    EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg']  #'-rgb.jpg' - includes normals-rgb.jpg
    EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr']
    EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr']
    EXT_MASK = ['-mask.png']
    for ext in EXT_COLOR_IMG:
        rgb_file_list += (sorted(glob.glob(os.path.join(dataset.image, '*' + ext))))
    for ext in EXT_DEPTH_IMG:
        depth_file_list += (sorted(glob.glob(os.path.join(dataset.depth, '*' + ext))))
    assert len(rgb_file_list) == len(depth_file_list), (
        'number of rgb ({}) and depth images ({}) are not equal'.format(len(rgb_file_list), len(depth_file_list)))

    if dataset.masks is not None and dataset.masks != '':
        for ext in EXT_MASK:
            segmentation_masks_list += (sorted(glob.glob(os.path.join(dataset.masks, '*' + ext))))
        assert len(rgb_file_list) == len(segmentation_masks_list), (
            'number of rgb ({}) and masks ({}) are not equal'.format(len(rgb_file_list),
                                                                        len(segmentation_masks_list)))
    if dataset.gt_depth is not None and dataset.gt_depth != '':
        for ext in EXT_DEPTH_GT:
            gt_depth_file_list += (sorted(glob.glob(os.path.join(dataset.gt_depth, '*' + ext))))
        assert len(rgb_file_list) == len(gt_depth_file_list), (
            'number of rgb ({}) and gt depth ({}) are not equal'.format(len(rgb_file_list),
                                                                        len(gt_depth_file_list)))

print('Total Num of rgb_files:', len(rgb_file_list))
print('Total Num of depth_files:', len(depth_file_list))
print('Total Num of gt_depth_files:', len(gt_depth_file_list))
print('Total Num of segmentation_masks:', len(segmentation_masks_list))
assert len(rgb_file_list) > 0, ('No files found in given directories')

# Create CSV File to store error metrics
csv_filename = 'computed_errors.csv'
field_names = ["Image Num", "RMSE", "REL", "MAE", "Delta 1.25", "Delta 1.25^2", "Delta 1.25^3"]
with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
    writer.writeheader()


for i in range(len(rgb_file_list)):

    # Run Depth Completion
    color_img = imageio.imread(rgb_file_list[i])
    input_depth = api_utils.exr_loader(depth_file_list[i], ndim=1)

    # NOTE: If no gt_depth present, it means the depth itself is gt_depth (syn data). We mask out all objects in input depth so depthcomplete can't cheat.
    if len(gt_depth_file_list) == 0 and len(segmentation_masks_list) > 0:
        if args.maskInputDepth:
            masks = imageio.imread(segmentation_masks_list[i])
            input_depth[masks > 0] = 0.0

    try:
        output_depth, filtered_output_depth = depthcomplete.depth_completion(
            color_img,
            input_depth,
            inertia_weight=float(config.depth2depth.inertia_weight),
            smoothness_weight=float(config.depth2depth.smoothness_weight),
            tangent_weight=float(config.depth2depth.tangent_weight),
            mode_modify_input_depth=config.modifyInputDepth.mode,
            dilate_mask=True)
    except depth_completion_api.DepthCompletionError as e:
        print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
        continue