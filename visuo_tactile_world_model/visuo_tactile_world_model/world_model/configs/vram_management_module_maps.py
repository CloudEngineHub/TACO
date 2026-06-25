VRAM_MANAGEMENT_MODULE_MAPS = {
    "visuo_tactile_world_model.world_model.model.wan.wan_video_dit.WanModel": {
        "visuo_tactile_world_model.world_model.model.wan.wan_video_dit.MLP": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_dit.DiTBlock": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedNonRecurseModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_dit.Head": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_dit.RMSNorm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
    },
    "visuo_tactile_world_model.world_model.model.wan.wan_video_text_encoder.WanTextEncoder": {
        "torch.nn.Linear": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_text_encoder.T5RelativeEmbedding": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_text_encoder.T5LayerNorm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
    },
    "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.WanVideoVAE": {
        "torch.nn.Linear": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.RMS_norm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.CausalConv3d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.Upsample": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.SiLU": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Dropout": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
    },
    "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.WanVideoVAE38": {
        "torch.nn.Linear": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.RMS_norm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.CausalConv3d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "visuo_tactile_world_model.world_model.model.wan.wan_video_vae.Upsample": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.SiLU": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Dropout": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
    },
    "visuo_tactile_world_model.world_model.model.wan.wan_video_image_encoder.WanImageEncoder": {
        "visuo_tactile_world_model.world_model.model.wan.wan_video_image_encoder.VisionTransformer": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "visuo_tactile_world_model.world_model.utils.vram.layers.AutoWrappedModule",
    },
}

VERSION_CHECKER_MAPS = {}
