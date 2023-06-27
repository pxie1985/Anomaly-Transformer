export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 1   --batch_size 256  --mode train --dataset Rain  --data_path data_for_model   --input_c 38  --output_c 38
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256     --mode test    --dataset Rain   --data_path dataset/SMD     --input_c 38  --output_c 38   --pretrained_model 20