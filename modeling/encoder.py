from timm.models.swin_transformer_v2 import SwinTransformerV2
import torch


class SwinTransformerEncoder(SwinTransformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pixels):
        x = self.forward_features(pixels)
        return x


if __name__ == '__main__':

    # encoder architecture config
    height, width = 128, 640
    channels = 1

    patch_size = 4
    window_size = 8

    embed_dim = 96 # will be multiplied by 4, will check later
    depths = [2, 6, 2]
    num_heads = [6, 12, 24]

    # create encoder
    swin_encoder = SwinTransformerEncoder(
        img_size=(height, width),
        patch_size=patch_size,
        in_chans=channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
    )

    # mandatory encoder inputs
    sample_pixels = torch.randn(2, channels, height, width)

    # encoder forward pass
    sample_outputs = swin_encoder(sample_pixels)
    print(sample_outputs.shape)
