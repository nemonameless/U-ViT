accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_huge.py

#accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth

