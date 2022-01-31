#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
python train.py --model_name="resnet18_cbam" \
                --image_size=64 \
                --num_classes=3 \
                --batch_size=4 \
                --num_epochs=200 \
                --model_save_path=test \
                --device=cpu \
                --pretrained_weights=False \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset="test_BUSI" \
                --num_gpus=0 \
                --dilute_mask=25 \
                --use_cbam=True \
                --use_mask=True \
                --no_channel=True \
                --reduction_ratio=16 \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                # --mask_annotate_file="data/mayo_patient_info.csv" \
                # --mask_annotate_dir="/Users/zongfan/Projects/data/breas_cancer_us/ultrasound/annotate"\