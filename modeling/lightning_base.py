import random
import pytorch_lightning as pl
import evaluate
import torch

CER = evaluate.load("cer")
ACC = evaluate.load("accuracy")


class LightningBase(pl.LightningModule):
    def __init__(self, tokenizer, model, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
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

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.global_step)

    def forward(self, features=None, target=None, mask=None):
        return self.model(
            features=features,
            target=target,
            mask=mask
        )

    def generate(self, seq_len=None, features=None, context=None):
        return self.model.generate(
            seq_len=seq_len,
            features=features,
            context=context,
        )

    def training_step(self, batch, batch_num):
        features = batch['features']
        target = batch['target']

        train_loss, _ = self(
            features=features,
            target=target,
        )

        self.log(
            name='train_loss', value=train_loss, on_step=True,
            on_epoch=False, prog_bar=True, logger=True
        )

        return train_loss

    def validation_step(self, batch, batch_num):
        features = batch['features']
        target = batch['target']

        val_loss, context = self(
            features=features,
            target=target,
        )

        generated = self.generate(
            context=context,
        )

        val_loss = val_loss.item()

        val_acc = ACC.compute(
            predictions=generated.flatten(),
            references=target.flatten(),
        )["accuracy"]

        generated = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True)
        target = self.tokenizer.batch_decode(
            target, skip_special_tokens=True)

        val_cer = CER.compute(
            predictions=generated,
            references=target,
        )

        return {
            'features': features,
            'target': target,
            'generated': generated,

            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_cer': val_cer,
        }

    def validation_epoch_end(self, outputs):

        wrong_cases = []
        right_cases = []

        for output in outputs:
            features = output['features']
            target = output['target']
            generated = output['generated']

            for i, (t, g) in enumerate(zip(target, generated)):
                if t != g:
                    wrong_cases.append((t, g))
                else:
                    right_cases.append(features[i])

        if len(wrong_cases) > 9:
            wrong_cases = random.sample(wrong_cases, 9)
        else:
            pass

        if len(right_cases) > 9:
            right_cases = random.sample(right_cases, 9)
        else:
            pass

        custom_log = '\n\n'.join(
            [f"- Ground Truth: {w1}\n- Prediction:&emsp;&ensp;{w2}" for w1, w2 in wrong_cases])

        self.logger.experiment.add_text(
            tag='wrong-cases', text_string=custom_log, global_step=self.global_step)

        for i, image in enumerate(right_cases):
            self.logger.experiment.add_image(
                tag=f"right-cases", img_tensor=image, global_step=self.global_step, dataformats="CHW")

        val_loss = sum([o['val_loss'] for o in outputs]) / len(outputs)
        val_acc = sum([o['val_acc'] for o in outputs]) / len(outputs)
        val_cer = sum([o['val_cer'] for o in outputs]) / len(outputs)

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
    target = torch.randint(0, 100, (2, 256))

    # forward pass
    loss, context = lightning_model(features, target)

    # generate
    generated = lightning_model.generate(
        seq_len=256,
        context=context,
    )

    print(loss)
    print(context.shape)
    print(generated.shape)
