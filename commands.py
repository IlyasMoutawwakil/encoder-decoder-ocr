import os
import torch
import pytorch_lightning as pl
from tokenization.tokenizer import CharacterTokenizer
from dataset.wikipedia_textline_dataset import WikipediaTextLineDataModule
from modeling.lightning_wrapper import VisionEncoderLanguageDecoderWrapper
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize

from configs import CHARACTERS, DATASET_NAME, ENCODER_CONFIG, DECODER_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG


def train_ocr(args):
    use_cuda = torch.cuda.is_available()
    
    max_train_steps = args.max_train_steps
    max_val_steps = args.max_val_steps

    batch_size = args.batch_size
    num_workers = args.num_workers

    language = args.language
    model_max_length = args.model_max_length

    characters = CHARACTERS[language]
    dataset_name = DATASET_NAME[language]

    encoder_name = args.encoder_name
    encoder_config = ENCODER_CONFIG[encoder_name]

    decoder_name = args.decoder_name
    decoder_config = DECODER_CONFIG[decoder_name]

    optimizer_name = args.optimizer_name
    optimizer_config = OPTIMIZER_CONFIG[optimizer_name]

    scheduler_name = args.scheduler_name
    scheduler_config = SCHEDULER_CONFIG[scheduler_name]

    experiment_name = f"ocr_{language}_{model_max_length}_{encoder_name}_{decoder_name}_{optimizer_name}_{scheduler_name}"

    if os.path.exists(f"checkpoints/{experiment_name}/"):
        print(
            f"Experiment {experiment_name} already exists, resuming training")
        ckpt_path = f"checkpoints/{experiment_name}/last.ckpt"
    else:
        print(f"Starting new experiment {experiment_name}")
        ckpt_path = None

    tokenizer = CharacterTokenizer(
        characters=characters,
        model_max_length=model_max_length,
    )

    simple_transform = Compose([
        Resize(
            (encoder_config["params"]["height"],
             encoder_config["params"]["width"])
        ),
        Grayscale(),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])

    datamodule = WikipediaTextLineDataModule(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        allowed_characters=characters,
        train_transform=simple_transform,
        val_transform=simple_transform,
    )

    model = VisionEncoderLanguageDecoderWrapper(
        tokenizer=tokenizer,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )

    prog_bar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=1,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"logs/{experiment_name}/",
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}/",
        filename="checkpoint-{epoch:03d}-{val_cer:.5f}",
        monitor="val_cer",
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="step",
    )

    trainer = pl.Trainer(
        accelerator='gpu' if use_cuda else 'cpu',

        max_epochs=-1,
        log_every_n_steps=1,
        num_sanity_val_steps=1,

        limit_val_batches=max_val_steps,
        limit_train_batches=max_train_steps,

        callbacks=[ckpt_callback, lr_monitor, prog_bar],
        enable_progress_bar=True,
        logger=logger,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
