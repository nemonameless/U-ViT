import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=500000,
        batch_size=256,
        log_interval=200,
        eval_interval=20000,
        save_interval=20000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_t2i',
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=1280, # gaint2 in unidiffuser 1536 30 24
        depth=30,
        num_heads=20,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77
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
        mini_batch_size=32, #50,
        cfg=True,
        scale=1.,
        path=''
    )

    return config
