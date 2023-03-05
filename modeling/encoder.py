from timm.models.swin_transformer_v2 import SwinTransformerV2
import torch


class SwinTransformerEncoder(SwinTransformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features):
        outputs = self.forward_features(features)
        return outputs


if __name__ == '__main__':

    # encoder architecture config
    height, width = 128, 640
    channels = 3

    patch_size = 4
    window_size = 8

    embed_dim = 96
    depths = [2, 6, 2]
    num_heads = [6, 12, 24]

    # create encoder
    encoder = SwinTransformerEncoder(
        img_size=(height, width),
        patch_size=patch_size,
        in_chans=channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
    )

    # mandatory encoder inputs
    sample_inputs = torch.randn(2, channels, height, width)

    # encoder forward pass
    sample_outputs = encoder(sample_inputs)
    print(sample_outputs.shape)
