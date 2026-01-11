#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -B ./tracking/track.py \
           --yolo-model yolo11x.pt \
           --reid-model osnet_x1_0_msmt17.pt \
           --tracking-method botsort \
           --source ./videos/3People_1264x720_20s.mp4 \
           --classes 0 \
           --custom-save-path ./runs/custom_track \
           --device 0
