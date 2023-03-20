import torch


class VisionEncoderLanguageDecoder(torch.nn.Module):
    def __init__(self, vision_encoder, language_decoder):

        super().__init__()

        self.vision_encoder = vision_encoder
        self.language_decoder = language_decoder

    def forward(
        self,
        pixels=None,
        input_tokens=None,
        memory_logits=None,
    ):

        if pixels is None and memory_logits is None:
            raise ValueError(
                "Either pixels or memory logits should be provided")

        elif memory_logits is None:
            memory_logits = self.vision_encoder(
                pixels=pixels,
            )

        output_logits = self.language_decoder(
            input_tokens=input_tokens,
            memory_logits=memory_logits,
        )

        return memory_logits, output_logits

    @torch.no_grad()
    def generate(
            self,
            seq_len=None,
            pixels=None,
            start_tokens=None,
            memory_logits=None,
    ):
        if pixels is None and memory_logits is None:
            raise ValueError(
                "Either pixels or memory logits should be provided")

        elif memory_logits is None:
            memory_logits = self.vision_encoder(
                pixels=pixels,
            )

        generated_tokens = self.language_decoder.generate(
            seq_len=seq_len,
            start_tokens=start_tokens,
            memory_logits=memory_logits,
        )

        return generated_tokens


if __name__ == '__main__':

    from encoder import SwinTransformerEncoder
    from decoder import AutoregressiveTransformerDecoder

    # encoder architecture config
    height, width = 128, 640
    channels = 3

    patch_size = 4
    window_size = 8

    embed_dim = 96
    depths = [2, 6, 2]
    num_heads = [6, 12, 24]

    # create encoder
    swin_encoder = SwinTransformerEncoder(
        height=height,
        width=width,
        in_chans=channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
    )

    # decoder config
    dim = 384
    heads = 8
    dropout = 0.1
    activation = 'gelu'
    norm_first = False  # norm before or after attention

    # decoder stack config
    num_layers = 4

    # language model config
    num_tokens = 256
    max_seq_len = 128

    # auto regression config
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    # autoregressive decoder
    autoregressive_decoder = AutoregressiveTransformerDecoder(
        dim=dim,
        heads=heads,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first,

        num_layers=num_layers,

        num_tokens=num_tokens,
        max_seq_len=max_seq_len,

        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # create vision encoder decoder
    vision_encoder_language_decoder = VisionEncoderLanguageDecoder(
        vision_encoder=swin_encoder,
        language_decoder=autoregressive_decoder,
    )

    # vision encoder decoder generation inputs
    sample_pixels = torch.randn(2, channels, height, width)
    sample_tokens = torch.randint(low=0, high=num_tokens, size=(2, 96))

    # vision encoder decoder forward pass
    memory_logits, output_logits = vision_encoder_language_decoder(
        pixels=sample_pixels,
        input_tokens=sample_tokens,
    )
    print(memory_logits.shape)
    print(output_logits.shape)

    # vision encoder decoder generation pass
    vision_encoder_language_decoder.eval()
    generated_tokens = vision_encoder_language_decoder.generate(
        pixels=sample_pixels
    )
    print(generated_tokens.shape)
