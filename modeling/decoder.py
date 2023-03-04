from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper, NonAutoregressiveWrapper


class AutoregressiveDecoder(AutoregressiveWrapper):
    def __init__(self,
                 num_tokens=1000,
                 max_seq_len=256,
                 pad_token_id=0,
                 decoder_config=dict(
                     dim=384,
                     depth=4,
                     heads=8,
                     cross_attend=True,
                     ff_glu=False,
                     attn_on_attn=False,
                     use_scalenorm=False,
                     rel_pos_bias=False
                 ),
                 *args,
                 **kwargs):

        net = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(**decoder_config)
        )

        super().__init__(net=net, pad_value=pad_token_id, *args, **kwargs)


if __name__ == '__main__':
    import torch
    from x_transformers.autoregressive_wrapper import top_k, top_p, top_a

    num_tokens = 1000
    max_seq_len = 256
    
    pad_token = 0

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

    auto_reg_decoder = AutoregressiveDecoder(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token,

        decoder_config=decoder_config
    )

    bos_token = 1
    eos_token = 2
    seq_len = 300

    generation_strategy = top_k  # top_k, top_p, top_a

    sample_context = torch.randn(1, 1, decoder_config['dim'])
    sample_start_tokens = torch.full((1, 1), bos_token)

    auto_reg_output = auto_reg_decoder.generate(
        start_tokens=sample_start_tokens,
        context=sample_context,

        seq_len=seq_len,
        eos_token=eos_token,

        filter_logits_fn=generation_strategy
    )

    print(auto_reg_output.shape)
