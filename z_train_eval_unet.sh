accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_t2i_unet_discrete.py --config=configs/mscoco_unet.py

#accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_t2i_unet_discrete.py --config=configs/mscoco_unet.py --nnet_path=

#python3.7 sample_t2i_discrete.py --config=configs/mscoco_unet.py --nnet_path=.pth --input_path=final.txt --output_path=mscoco_unet_t2i_vis/
