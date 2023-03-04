import torch


class VisionEncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, target, mask=None):

        context = self.encoder(features)

        loss = self.decoder(
            target,
            mask=mask,
            context=context
        )

        return loss, context

    @torch.no_grad()
    def generate(
            self,
            seq_len,
            bos_token_id,
            eos_token_id,
            features=None,
            context=None,

    ):

        if features is None and context is None:
            raise ValueError(
                "Either features or context should be provided")

        if context is None:
            context = self.encoder(features)

        start_tokens = torch.full(
            fill_value=bos_token_id,
            size=(features.size(0), 1),
            device=features.device,
        )

        generated = self.decoder.generate(
            start_tokens=start_tokens,
            context=context,

            seq_len=seq_len,
            eos_token=eos_token_id,
        )

        return generated


if __name__ == '__main__':

    import torch

    from encoder import SwinTransformerEncoder
    from decoder import AutoregressiveDecoder

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

    decoder_config = dict(
        dim=384,
        depth=4,
        heads=8,
        cross_attend=True,
        ff_glu=False,
        attn_on_attn=False,
        use_scalenorm=False,
        rel_pos_bias=False
    )

    num_tokens = 100
    max_seq_len = 256
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    seq_len = 10

    decoder = AutoregressiveDecoder(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,

        decoder_config=decoder_config
    )

    model = VisionEncoderDecoder(
        encoder=swin_encoder,
        decoder=decoder,
    )

    features = torch.randn(1, channels, height, width)
    target = torch.randint(0, seq_len, (1, max_seq_len))

    loss, context = model(features, target)
    print(loss)

    generated = model.generate(
        seq_len,
        bos_token_id,
        eos_token_id,
        features=features
    )

    print(generated)
