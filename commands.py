import os
import torch
import pytorch_lightning as pl
from tokenization.tokenizer import CharacterTokenizer
from dataset.wikipedia_dataset import WikipediaTextLineDataModule
from modeling.encoder_decoder import VisionEncoderLanguageDecoder
from modeling.lightning_wrapper import VisionEncoderLanguageDecoderWrapper
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize

from configs import CHARACTERS, DATASET_NAME, ENCODER_CONFIG, DECODER_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG


def train_ocr(args):
    use_cuda = torch.cuda.is_available()
    max_train_steps = args.max_train_steps
    max_val_steps = args.max_val_steps
    language = args.language
    characters = CHARACTERS[language]
    model_max_length = args.model_max_length

    dataset_name = DATASET_NAME[language]
    batch_size = args.batch_size
    num_workers = args.num_workers

    encoder_name = args.encoder_name
    encoder_config = ENCODER_CONFIG[encoder_name]

    decoder_name = args.decoder_name
    decoder_config = DECODER_CONFIG[decoder_name]

    optimizer_name = args.optimizer_name
    optimizer_config = OPTIMIZER_CONFIG[optimizer_name]

    scheduler_name = args.scheduler_name
    scheduler_config = SCHEDULER_CONFIG[scheduler_name]

    tokenizer = CharacterTokenizer(
        characters=characters,
        model_max_length=model_max_length,
    )
    transform = Compose([
        Resize((encoder_config["params"]["height"],
               encoder_config["params"]["width"])),
        Grayscale(),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    datamodule = WikipediaTextLineDataModule(
        dataset_name=dataset_name,
        transform=transform,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        characters=characters,
    )

    visiion_encoder = encoder_config["class"](
        **encoder_config["params"]
    )

    language_decoder = decoder_config["class"](
        num_tokens=len(characters) + 4,
        max_seq_len=model_max_length,
        **decoder_config["params"]
    )

    visiion_encoder_language_decoder = VisionEncoderLanguageDecoder(
        vision_encoder=visiion_encoder,
        language_decoder=language_decoder,
    )

    # optimizer config

    experiment_name = f"ocr_{language}_{encoder_name}_{decoder_name}_{optimizer_name}_{scheduler_name}_{model_max_length}"

    if os.path.exists(f"checkpoints/{experiment_name}/"):
        print(
            f"Experiment {experiment_name} already exists, resuming training")
        ckpt = torch.load(f"checkpoints/{experiment_name}/last.ckpt")
    else:
        print(f"Starting new experiment {experiment_name}")
        ckpt = None

    lightning_model = VisionEncoderLanguageDecoderWrapper(
        model=visiion_encoder_language_decoder,
        tokenizer=tokenizer,
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
        accelerator="gpu" if use_cuda else 'cpu',

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
        model=lightning_model,
        datamodule=datamodule,
        ckpt_path=ckpt,
    )
