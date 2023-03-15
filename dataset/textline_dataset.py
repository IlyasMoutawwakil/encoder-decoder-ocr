from torchvision.datasets import VisionDataset
import random
import torch

try:
    from dataset.text_cleaning_utils import get_random_cut
    from dataset.line_generation_utils import generate_line
except:
    from text_cleaning_utils import get_random_cut
    from line_generation_utils import generate_line

# expected format of the dataset
DUMMY_DATASET = [
    {
        "lines": [
            f"Ceci est un test ààà {i}", 
            f"Ceci est un test ççç {i}", 
            f"Ceci est un test ééé {i}"
        ],
    } for i in range(100)
]

class TextLineDataset(VisionDataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            transform,
    ):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = random.choice(self.dataset[idx]['lines'])   
        line = get_random_cut(
            line,
            # - 2 for the [BOS] and [EOS] tokens
            max_length=self.tokenizer.model_max_length - 2,
        )
        target = self.tokenizer.encode(
            line,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        ).squeeze(0)

        try:
            image = generate_line(line)
        except Exception as e:
            return self.__getitem__(idx)

        features = self.transform(image)

        return {
            "features": features,
            "target": target,
        }


if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    
    # create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased", 
        model_max_length=100
    )
    
    # create a transform
    transform = Compose([
        Resize((128, 640)),
        Grayscale(),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])

    # create a dataset
    dataset = TextLineDataset(
        dataset=DUMMY_DATASET,
        tokenizer=tokenizer,
        transform=transform,
    )

    # get a sample from the dataset
    sample = dataset[0]

    # print the shapes and dtypes of the sample
    print(sample["features"].shape, sample["features"].dtype)
    print(sample["target"].shape, sample["target"].dtype)
    print(sample["target"])
    # print(sample["mask"].shape, sample["mask"].dtype)

    # show the images and the text
    for i in range(10):
        # get a sample from the dataset
        sample = dataset[i]
        print(tokenizer.decode(sample["target"], skip_special_tokens=True))
        plt.imshow(sample["features"].permute(1, 2, 0))
        plt.show()
        