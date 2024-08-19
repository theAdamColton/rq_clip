rank="$1"

CUDA_VISIBLE_DEVICES=0 python3 src/preprocess_a6000.py -r "$rank" -b 75