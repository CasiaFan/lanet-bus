#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
python train.py --model_name=resnet50 \
                --input_size=448 \
                --num_classes=3 \
                --batch_size=36 \
                --num_epochs=150 \
                --model_save_path=BUSI_res50 \
                --device=cuda:0 \
                --lr=0.001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset="BUSI" \
                --num_gpus=2
