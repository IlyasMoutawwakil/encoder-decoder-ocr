import pytorch_lightning as pl
import evaluate
import torch
import timm

CER = evaluate.load("cer")
ACC = evaluate.load("accuracy")


class LightningBase(pl.LightningModule):
    def __init__(self, model, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):

        optimizer_class = self.optimizer_config['base_class']
        optimizer = optimizer_class(
            params=self.parameters(),
            **self.optimizer_config['params'],
        )

        scheduler_class = self.scheduler_config['base_class']
        scheduler = scheduler_class(
            optimizer=optimizer,
            **self.scheduler_config['params'],
        )

        return [optimizer], [scheduler]

    def forward(self, features, targets, mask=None):
        return self.model(
            features=features,
            targets=targets,
            mask=mask
        )

    def generate(self, seq_len, features=None, context=None):
        return self.model.generate(
            seq_len=seq_len,
            features=features,
            context=context,
        )

    def training_step(self, batch, batch_num):
        features, (targets, masks) = batch

        train_loss, context = self(
            features=features,
            targets=targets,
            mask=masks
        )

        self.log(
            name='train_loss', value=train_loss, on_step=True,
            on_epoch=False, prog_bar=True, logger=True
        )

        return train_loss

    def validation_step(self, batch, batch_num):
        features, (targets, masks) = batch

        val_loss, context = self(
            features=features,
            targets=targets,
            mask=masks
        )

        seq_len = self.model.decoder.max_seq_len
        generated = self.generate(
            seq_len=seq_len,
            context=context,
        )

        # pad generated and targets to the same length
        max_len = max(generated.shape[1], targets.shape[1])
        generated = torch.nn.functional.pad(
            input=generated,
            pad=(0, max_len - generated.shape[1]),
            value=self.model.decoder.pad_token_id
        )

        targets = torch.nn.functional.pad(
            input=targets,
            pad=(0, max_len - targets.shape[1]),
            value=self.model.decoder.pad_token_id
        )

        val_acc = ACC.compute(
            predictions=generated,
            references=targets,
        )

        val_cer = CER.compute(
            predictions=self.tokenizer.decode(generated),
            references=self.tokenizer.decode(targets),
        )

        return {
            'features': features,
            'targets': targets,
            'generated': generated,
            
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_cer': val_cer,
        }

    def validation_epoch_end(self, outputs):

        features = torch.cat([o['features'] for o in outputs], dim=0)
        targets = torch.cat([o['targets'] for o in outputs], dim=0)
        generated = torch.cat([o['generated'] for o in outputs], dim=0)

        val_loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_acc = torch.stack([o['val_acc'] for o in outputs]).mean()
        val_cer = torch.stack([o['val_cer'] for o in outputs]).mean()

        self.log(
            name='val_loss', value=val_loss, on_step=False,
            on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            name='val_acc', value=val_acc, on_step=False,
            on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            name='val_cer', value=val_cer, on_step=False,
            on_epoch=True, prog_bar=True, logger=True
        )

        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_cer': val_cer,
        }


if __name__ == '__main__':

    from encoder import SwinTransformerEncoder
    from decoder import AutoregressiveDecoder
    from vision_encoder_decoder import VisionEncoderDecoder

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

    # optimizer config
    from timm.optim import AdamW
    optimizer_config = dict(
        base_class=AdamW,
        params=dict(
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
        ),
    )

    # schedueler config
    from timm.scheduler.cosine_lr import CosineLRScheduler
    scheduler_config = dict(
        base_class=CosineLRScheduler,
        params=dict(
            t_initial=500,
            lr_min=1e-6,
            cycle_mul=3,
            cycle_decay=0.8,
            cycle_limit=20,
            warmup_t=20,
            k_decay=1.5,
        ),
    )

    # create lightning model
    lightning_model = LightningBase(
        model=model,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )

    # create one batch
    features = torch.randn(2, 3, 128, 640)
    targets = torch.randint(0, 100, (2, 256))
    masks = None

    # forward pass
    loss, context = lightning_model(features, targets, masks)

    # generate
    generated = lightning_model.generate(
        seq_len=256,
        context=context,
    )
    
    print(loss)
    print(context.shape)
    print(generated.shape)