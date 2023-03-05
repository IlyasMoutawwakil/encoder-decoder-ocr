import torch


class VisionEncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, targets, mask=None):
        context = self.encoder(features)

        loss = self.decoder(
            targets,
            mask=mask,
            context=context,
        )

        return loss, context

    @torch.no_grad()
    def generate(
            self,
            seq_len,
            features=None,
            context=None,
            *args,
            **kwargs
    ):
        if features is None and context is None:
            raise ValueError(
                "Either features or context should be provided")

        if context is None:
            context = self.encoder(features)

        generated = self.decoder.generate(
            seq_len=seq_len,
            context=context,
            *args,
            **kwargs
        )

        return generated


if __name__ == '__main__':

    from encoder import SwinTransformerEncoder
    from decoder import AutoregressiveDecoder

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

    # decoder architecture config
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

    # auto regressive wrapper architecture config
    num_tokens = 100
    max_seq_len = 256

    # auto regressive wrapper generation config
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    # create decoder
    decoder = AutoregressiveDecoder(
        decoder_config=decoder_config,

        num_tokens=num_tokens,
        max_seq_len=max_seq_len,

        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # create vision encoder decoder
    model = VisionEncoderDecoder(
        encoder=encoder,
        decoder=decoder,
    )

    # vision encoder decoder generation inputs
    seq_len = 96
    features = torch.randn(2, channels, height, width)
    targets = torch.randint(0, num_tokens, (2, seq_len))

    # vision encoder decoder forward pass
    loss, context = model(
        features=features,
        targets=targets
    )
    print(loss, context.shape)

    # vision encoder decoder generation pass
    generated = model.generate(
        seq_len=seq_len,
        features=features
    )
    print(generated.shape, targets.shape)
