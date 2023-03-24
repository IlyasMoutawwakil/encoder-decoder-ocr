import pytorch_lightning as pl
import torchmetrics
import random
import torch


class VisionEncoderLanguageDecoderWrapper(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        encoder_config,
        decoder_config,
        optimizer_config,
        scheduler_config,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer

        self.vision_encoder = encoder_config["class"](
            **encoder_config["params"]
        )

        self.language_decoder = decoder_config["class"](
            num_tokens=self.tokenizer.vocab_size,
            max_seq_len=self.tokenizer.model_max_length,
            **decoder_config["params"]
        )

        self.optimizer = optimizer_config['class'](
            self.parameters(),
            **optimizer_config['params'],
        )
        self.scheduler = scheduler_config['class'](
            self.optimizer,
            **scheduler_config['params'],
        )

        self.save_hyperparameters()

        self.cer = torchmetrics.CharErrorRate()

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

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

        self.cer(generated_tokens, tokens)
        self.log(
            name='val_cer', value=self.cer, on_step=False,
            on_epoch=True, prog_bar=True, logger=True
        )

        if batch_num == 0:
            self.outputs = {
                'pixels': [],
                'true_strings': [],
                'generated_strings': []
            }

        self.outputs['pixels'].append(pixels)
        self.outputs['true_strings'].append(true_strings)
        self.outputs['generated_strings'].append(generated_strings)

    def on_validation_epoch_end(self):

        wrong_cases = []
        right_cases = []

        for i in range(len(self.outputs['pixels'])):
            if self.outputs['generated_strings'][i] != self.outputs['true_strings'][i]:
                wrong_cases.append(
                    (self.outputs['true_strings'][i],
                     self.outputs['generated_strings'][i])
                )
            else:
                right_cases.append(self.outputs['pixels'][i])

        if len(wrong_cases) > 9:
            wrong_cases = random.sample(wrong_cases, 9)

        if len(right_cases) > 9:
            right_cases = random.sample(right_cases, 9)

        custom_log = '\n\n'.join(
            [
                f"- Ground Truth: {w1}\n- Prediction:&emsp;&ensp;{w2}"
                for w1, w2 in wrong_cases
            ]
        )

        self.logger.experiment.add_text(
            tag='wrong-cases',
            text_string=custom_log,
            global_step=self.global_step
        )

        for i, image in enumerate(right_cases):
            self.logger.experiment.add_image(
                tag=f"right-cases",
                img_tensor=image,
                global_step=self.global_step,
                dataformats="CHW"
            )

        self.outputs = {
            'pixels': [],
            'true_strings': [],
            'generated_strings': []
        }
