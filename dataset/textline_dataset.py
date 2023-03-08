from torchvision.datasets import VisionDataset
import numpy as np
import random
import torch

from dataset.text_cleaning_utils import get_random_cut
from dataset.line_generation_utils import generate_line


class TextLineDataset(VisionDataset):
    def __init__(
            self,
            tokenizer,
            dataset,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = random.choice(self.dataset[idx]['lines'])   
        line = get_random_cut(
            line,
            # - 2 for the [BOS] and [EOS] tokens
            max_length=self.tokenizer.model_max_length - 2,
        )
        tokens = self.tokenizer(
            line,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        target = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)

        try:
            image = generate_line(line)
        except Exception as e:
            return self.__getitem__(idx)

        features = torch.tensor(
            np.array(image)
        ).permute(2, 0, 1) / 127.5 - 1

        return {
            "features": features,
            "target": target,
            "mask": mask,
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    
    dataset = [
        {
            "lines": ["Ceci est un test ààà 1", "Ceci est un test ççç 2", "Ceci est un test éééé 3", ],
        }
    ] * 100
    
    # get a generic tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased", 
        model_max_length=100
    )

    # create a dataset
    dataset = TextLineDataset(
        tokenizer=tokenizer,
        dataset=dataset,

    )

    # get a sample from the dataset
    sample = dataset[55]

    # print the shapes and dtypes of the sample
    print(sample["features"].shape, sample["features"].dtype)
    print(sample["target"].shape, sample["target"].dtype)
    print(sample["mask"].shape, sample["mask"].dtype)

    # show the image

    for i in range(10):
        # get a sample from the dataset
        sample = dataset[i]
        print(tokenizer.decode(sample["target"], skip_special_tokens=True))
        plt.imshow((sample["features"].permute(1, 2, 0) + 1) / 2)
        plt.show()
        