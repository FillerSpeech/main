# # # # # # # # # # # # # # # # # # # # # Filler prediction inference # # # # # # # # # # # # # # # # # # # #

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/test_config_filler_pred_1.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/test_config_filler_pred_2.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/test_config_filler_pred_3.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./train.py --cfg-path ./configs/test_config_filler_pred_4.yaml

