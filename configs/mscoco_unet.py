import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 32, 32) # (4, 64, 64)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=500000,
        batch_size=16,
        log_interval=200,
        eval_interval=20000,
        save_interval=20000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.01, #0.03,
        betas=(0.9, 0.999), #
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='unet',
        sample_size=32, # img_size
        in_channels=4,
        out_channels=4,
        down_block_types=['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2,
        cross_attention_dim=1280,
        # clip_dim=768,
        # num_clip_token=77,
    )

    config.nnet_tiny = d(
        name='unet',
        sample_size=32, # img_size
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=[32, 64],
        layers_per_block=2,
        cross_attention_dim=32,
        # clip_dim=768,
        # num_clip_token=77,
    )

    config.nnet2 = d(
        name='unet',
        act_fn="silu",
        attention_head_dim=8,
        block_out_channels=[320, 640, 1280, 1280],
        center_input_sample=False,
        cross_attention_dim=1280,
        down_block_types=['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
        downsample_padding=1,
        flip_sin_to_cos=True,
        freq_shift=0,
        in_channels=4,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-05,
        norm_num_groups=32,
        out_channels=4, #
        sample_size=32, # img_size
        up_block_types=['UpBlock2D', "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        # clip_dim=768,
        # num_clip_token=77,
    )

    config.dataset = d(
        name='mscoco256_features',
        path='assets/datasets/coco256_features',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=30000,
        mini_batch_size=32, #
        cfg=True,
        scale=1.,
        path=''
    )

    return config
