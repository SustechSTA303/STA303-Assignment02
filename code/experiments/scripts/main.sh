export CUDA_VISIBLE_DEVICES=0

VISUAL_BACKBONE=('RN50' 'RN101' 'ViT-B/16' 'ViT-B/32')
# VISUAL_BACKBONE=('ViT-B/16' 'ViT-B/32')
ROUND=(1 2 3 4 5 6 7 8 9 10)
# NUM_CALIB=(100 500 1000 1500 2000)
NUM_CALIB=(10 50)
# NUM_CALIB=(5000)

for visual_backbone in "${VISUAL_BACKBONE[@]}"; do
    for num_calib in "${NUM_CALIB[@]}"; do
        for round in "${ROUND[@]}"; do
            echo "Backbone:${visual_backbone}"
            python main.py --backbone "${visual_backbone}" --num_calib "${num_calib}"
        done
    done
done