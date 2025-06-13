# # # # # # # # # # # # # # # # # # # # # Filler prediction Training # # # # # # # # # # # # # # # # # # # #

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./filler_prediction/train.py --cfg-path ./filler_prediction/configs/config_filler_pred.yaml

