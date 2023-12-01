#!/bin/bash

# VISUAL_BACKBONES=(RN50 ViT-B/32 ViT-B/16)
VISUAL_BACKBONES=(ViT-B/32 ViT-B/16)
PERCENTAGE=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 90 95)

for VB in "${VISUAL_BACKBONES[@]}"
do
    for PER in "${PERCENTAGE[@]}"
    do
        echo ==========================+++++++++++++++++++++++++++
        echo VB: $VB, PER: $PER
        echo msp.py --vb $VB --percentage $PER
        echo ==========================+++++++++++++++++++++++++++
        python msp.py --vb $VB --percentage $PER --cuda 5
    done
done