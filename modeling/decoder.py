import torch

class AutoregressiveTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        # decoder architecture config
        dim=384,
        heads=8,
        dropout=0.1,
        activation='gelu',
        norm_first=True,
        # decoder stack config
        num_layers=4,
        # language model head config
        num_tokens=256,
        max_seq_len=128,
        # auto regression config
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):

        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        self.num_layers = num_layers

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.tok_embed = torch.nn.Embedding(num_tokens, dim)
        self.pos_embed = torch.nn.Embedding(max_seq_len, dim)
        self.embed_drop = torch.nn.Dropout(dropout)

        self.decs_stak = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        self.final_norm = torch.nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        self.lang_mod_head = torch.nn.Linear(dim, num_tokens, bias=False)

        # some optimizations I found in the nanoGPT repo
        self.lang_mod_head.weight = self.tok_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            if module.weight.size(-1) == self.dim * 4:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/(2 * self.num_layers)**0.5)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                module.bias = None

    def forward(
        self,
        input_tokens,
        memory_logits=None,
    ):

        seq_len = input_tokens.size(1)

        assert seq_len <= self.max_seq_len, \
            f"Cannot forward sequence of length {seq_len}," \
            f"when max sequence length is only {self.max_seq_len}"

        if memory_logits is None:
            # decoder only mode
            memory_logits = x

        positions = torch.arange(
            start=0,
            end=seq_len,
            device=input_tokens.device,
        ).unsqueeze(0)

        t_e = self.tok_embed(input_tokens)
        p_e = self.tok_embed(positions)
        x = self.embed_drop(t_e + p_e)

        for dec_layer in self.decs_stak:
            x = dec_layer(
                tgt=x,
                memory=memory_logits,
                tgt_is_causal=True,
                memory_is_causal=False,
            )

        x = self.final_norm(x)
        
        if self.training:
            # if we are training, we predict all the tokens
            logits = self.lang_mod_head(x)
        else:
            # if we are in inference (generation),
            # we only predict the last token
            logits = self.lang_mod_head(x[:, [-1], :])

        return logits

    @torch.no_grad()
    def generate(
        self,
        seq_len=None,
        start_tokens=None,
        memory_logits=None,
        temperature=1.,
        top_k=None,
    ):

        assert self.training is False, \
            "Generation is only available in inference mode"

        if start_tokens is not None:
            # start with the provided tokens
            generated_tokens = start_tokens
        elif memory_logits is not None:
            # start with the bos token
            batch_size = memory_logits.size(0)
            generated_tokens = torch.full(
                size=(batch_size, 1),
                fill_value=self.bos_token_id,
                device=memory_logits.device,
            )
        else:
            raise ValueError(
                "Either start tokens or memory logits must be provided"
                "to infer batch size."
            )

        if seq_len is None:
            seq_len = self.max_seq_len

        for _ in range(seq_len-1):

            if memory_logits is None:
                # decoder only mode
                memory_logits = generated_tokens

            # only need the last token predictions (damn autoregressive models)
            logits = self(
                generated_tokens,
                memory_logits=memory_logits,
            )

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                # only keep top k tokens
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)

            # sample from the distribution
            next_tokens = torch.multinomial(probs, num_samples=1)

            # append the new tokens to the generated sequence
            generated_tokens = torch.cat(
                [generated_tokens, next_tokens], dim=1)

            # check if sentences have ended
            is_eos_tokens = generated_tokens == self.eos_token_id
            if is_eos_tokens.any(dim=-1).all():
                # mask/pad all tokens after the first eos or pad token
                shifted_is_eos_tokens = torch.nn.functional.pad(
                    is_eos_tokens,
                    pad=(1, -1),
                )
                mask = shifted_is_eos_tokens.cumsum(dim=-1) >= 1
                generated_tokens = generated_tokens.masked_fill(
                    mask,
                    value=self.pad_token_id
                )

                # pad all sequences to the max length for consistency
                generated_tokens = torch.nn.functional.pad(
                    generated_tokens,
                    (0, seq_len - generated_tokens.size(1)),
                    value=self.pad_token_id
                )
                break

        return generated_tokens


if __name__ == '__main__':

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
        num_layers=num_layers,

        dim=dim,
        heads=heads,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first,

        num_tokens=num_tokens,
        max_seq_len=max_seq_len,

        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # mandatory decoder inputs
    sample_tokens = torch.randint(low=0, high=num_tokens, size=(2, 96))
    sample_memory = torch.randn(2, 64, 384)

    # auto regressive wrapper forward pass
    logits = autoregressive_decoder(
        input_tokens=sample_tokens,
        memory_logits=sample_memory,
    )
    print(logits.shape)

    # generation pass
    autoregressive_decoder.eval()
    sample_outputs = autoregressive_decoder.generate(
        memory_logits=sample_memory,
    )
    print(sample_outputs)
    print(sample_outputs.shape)
