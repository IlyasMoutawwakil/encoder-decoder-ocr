from torchvision.datasets import VisionDataset
import numpy as np
import torch

from text_cleaning_utils import get_random_cut, clean_hf_dataset
from line_generation_utils import generate_line


class TextLinesDataset(VisionDataset):
    def __init__(
            self,
            tokenizer,
            paragraphs,
    ):
        self.tokenizer = tokenizer
        self.paragraphs = paragraphs

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]['paragraph']
        
        text = get_random_cut(
            paragraph,
            max_length=self.tokenizer.model_max_length - 2, # -2 for the [BOS] and [EOS] tokens
        )
        
        image = generate_line(text)
        
        features = torch.tensor(
            np.array(image)
        ).permute(2, 0, 1) / 127.5 - 1
                
        tokens = self.tokenizer(
            text,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        target = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)

        return {
            "features": features,
            "target": target,
            "mask": mask,
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    import datasets as ds
    
    paragraphs_dataset = ds.load_dataset(
        "asi/wikitext_fr", 
        "wikitext-72",
        split="train"
    )
    
    paragraphs_dataset = clean_hf_dataset(
        paragraphs_dataset
    )
    
    # get a generic tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased", 
        model_max_length=100
    )

    # create a dataset
    dataset = TextLinesDataset(
        tokenizer=tokenizer,
        paragraphs=paragraphs_dataset,

    )

    # get a sample from the dataset
    sample = dataset[55]

    # print the shapes and dtypes of the sample
    print(sample["features"].shape, sample["features"].dtype)
    print(sample["target"].shape, sample["target"].dtype)
    print(sample["mask"].shape, sample["mask"].dtype)

    # decode the target
    print(tokenizer.decode(sample["target"], skip_special_tokens=True))

    # show the image

    for i in range(10):
        # get a sample from the dataset
        sample = dataset[i]
        plt.imshow((sample["features"].permute(1, 2, 0) + 1) / 2)
        plt.show()
        