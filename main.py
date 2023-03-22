import argparse
from commands import train_ocr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train_ocr"])

    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="The maximum number of training steps",
        default=1000,
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        help="The maximum number of validation steps",
        default=10,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of threads to use " "for loading the data",
        default=8,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The size of a minibatch",
        default=16,
    )
    
    parser.add_argument(
        "--model_max_length",
        type=int,
        help="The maximum length of the model",
        default=128,
    )

    parser.add_argument(
        "--language",
        type=str,
        help="The language to train on",
        default="fr",
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        help="The name of the encoder to use",
        default="SwinTransformerEncoder",
    )
    parser.add_argument(
        "--decoder_name",
        type=str,
        help="The name of the decoder to use",
        default="AutoregressiveTransformerDecoder",
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        help="The name of the optimizer to use",
        default="AdamW",
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        help="The name of the scheduler to use",
        default="CosineLRScheduler",
    )

    args = parser.parse_args()

    if args.command == "train_ocr":
        train_ocr(args)

    else:
        raise ValueError("Unknown command")
