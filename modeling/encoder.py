from timm.models.swin_transformer_v2 import SwinTransformerV2


class SwinTransformerEncoder(SwinTransformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


if __name__ == '__main__':
    import torch

    height, width = 128, 640
    channels = 3

    patch_size = 4
    window_size = 8

    embed_dim = 96
    depths = [2, 6, 2]
    num_heads = [6, 12, 24]

    swin_encoder = SwinTransformerEncoder(
        img_size=(height, width),
        patch_size=patch_size,
        in_chans=channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
    )

    sample_input = torch.randn(1, channels, height, width)
    sample_output = swin_encoder(sample_input)

    print(sample_output.shape)
