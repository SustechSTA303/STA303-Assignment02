VISUAL_BACKBONES=(RN50 ViT-B/32 ViT-B/16)

for VB in "${VISUAL_BACKBONES[@]}"
do
    echo ==========================+++++++++++++++++++++++++++
    echo VB: $VB, PER: 85
    echo msp.py --vb $VB --percentage 85
    echo ==========================+++++++++++++++++++++++++++
    python msp.py --vb $VB --percentage 85 --cuda 5
done