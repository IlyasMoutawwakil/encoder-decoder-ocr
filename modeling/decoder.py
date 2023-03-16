from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper, NonAutoregressiveWrapper
import torch


class AutoregressiveDecoder(AutoregressiveWrapper):
    def __init__(self,
                 decoder_config=dict(
                     dim=384,
                     depth=4,
                     heads=8,
                     cross_attend=True,  # enables cross attention using the context argument, default is True
                     ff_glu=False,  # use Gated Linear Units in the feed forward layer, default is False
                     attn_on_attn=False,  # use attention on attention, default is False
                     use_scalenorm=False,  # use ScaleNorm instead of LayerNorm, default is False
                     rel_pos_bias=False,  # use relative positional bias, default is False
                 ),

                 num_tokens=100,
                 max_seq_len=256,

                 bos_token_id=0,
                 eos_token_id=1,
                 pad_token_id=2,

                 **kwargs):

        net = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(**decoder_config)
        )

        super().__init__(
            net=net,
            pad_value=pad_token_id,
            **kwargs,
        )

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def forward(self, target, **kwargs):

        return super().forward(
            target,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, seq_len=None, context=None, **kwargs):

        if seq_len is None:
            seq_len = self.max_seq_len

        if context is None:
            raise ValueError('context must be provided to generate')

        start_tokens = torch.full(
            (context.size(0), 1),
            fill_value=self.bos_token_id,
            device=context.device
        )

        return super().generate(
            seq_len=seq_len,
            start_tokens=start_tokens,
            eos_token=self.eos_token_id,
            context=context,
            **kwargs,
        )


if __name__ == '__main__':

    # decoder architecture config
    decoder_config = dict(
        dim=384,
        depth=4,
        heads=8,
        cross_attend=True,
        ff_glu=False,
        attn_on_attn=False,
        use_scalenorm=False,
        rel_pos_bias=False,
    )

    # auto regressive wrapper architecture config
    num_tokens = 256
    max_seq_len = 128

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

    # mandatory decoder inputs
    seq_len = 96
    sample_context = torch.randn(2, 320, 384)
    sample_target = torch.randint(0, num_tokens, (2, seq_len))

    # auto regressive wrapper forward pass
    loss = decoder(
        target=sample_target,
        context=sample_context,
    )
    print(loss)

    # optional generation inputs
    from x_transformers.autoregressive_wrapper import top_k
    optional_inputs = dict(
        filter_logits_fn=top_k,
        temperature=1.0,
    )

    # generation pass
    sample_outputs = decoder.generate(
        seq_len=seq_len,
        context=sample_context,

        **optional_inputs
    )
    print(sample_outputs.shape)
