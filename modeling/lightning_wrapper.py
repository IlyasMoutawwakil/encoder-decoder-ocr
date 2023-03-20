from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import AdamW

import pytorch_lightning as pl
import evaluate
import random
import torch

CER = evaluate.load("cer")


class VisionEncoderLanguageDecoderWrapper(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizer_config, scheduler_config, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.tokenizer = tokenizer

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.outputs = []

    def configure_optimizers(self):

        optimizer = self.optimizer_config['class'](
            self.parameters(),
            **self.optimizer_config['params'],
        )

        scheduler = self.scheduler_config['class'](
            optimizer,
            **self.scheduler_config['params'],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.global_step)

    def forward(
        self,
        pixels=None,
        input_tokens=None,
        memory_logits=None,
    ):
        return self.model(
            pixels=pixels,
            input_tokens=input_tokens,
            memory_logits=memory_logits,
        )

    def generate(
        self,
        seq_len=None,
        pixels=None,
        start_tokens=None,
        memory_logits=None,
    ):
        return self.model.generate(
            seq_len=seq_len,
            pixels=pixels,
            start_tokens=start_tokens,
            memory_logits=memory_logits,
        )

    def training_step(self, batch, batch_num):
        pixels = batch['pixels']
        tokens = batch['tokens']

        input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]

        memory_logits, output_logits = self(
            pixels=pixels,
            input_tokens=input_tokens,
        )

        train_loss = torch.nn.functional.cross_entropy(
            output_logits.flatten(0, 1),
            target_tokens.flatten(0, 1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log(
            name='train_loss', value=train_loss, on_step=True,
            on_epoch=False, prog_bar=True, logger=True
        )

        return train_loss

    def validation_step(self, batch, batch_num):
        pixels = batch['pixels']
        tokens = batch['tokens']
        seq_len = tokens.size(1)

        generated_tokens = self.generate(
            seq_len=seq_len,
            pixels=pixels,
        )

        generated_strings = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        true_strings = self.tokenizer.batch_decode(
            tokens, skip_special_tokens=True)

        val_cer = CER.compute(
            predictions=generated_strings,
            references=true_strings,
        )

        self.outputs.append({
            'pixels': pixels,
            'true_strings': true_strings,
            'generated_strings': generated_strings,

            'val_cer': val_cer,
        })

    def on_validation_epoch_end(self):

        wrong_cases = []
        right_cases = []

        for output in self.outputs:
            pixels = output['pixels']
            true_strings = output['true_strings']
            generated_strings = output['generated_strings']

            for i, (t, g) in enumerate(zip(true_strings, generated_strings)):
                if t != g:
                    wrong_cases.append((t, g))
                else:
                    right_cases.append(pixels[i])

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

        val_cer = sum([o['val_cer'] for o in self.outputs]) / len(self.outputs)

        self.log(
            name='val_cer', value=val_cer, on_step=False,
            on_epoch=True, prog_bar=True, logger=True
        )

        self.outputs = []

        return {
            'val_cer': val_cer,
        }
