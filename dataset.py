import torch
import numpy as np
from captcha.image import ImageCaptcha
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import string

background_char = "-"
characters = [
    i
    for i in (
        background_char
        + string.digits
        + string.ascii_lowercase
        + string.ascii_uppercase
    )
]
char_pos = {c: i for i, c in enumerate(characters)}
num_classes = len(characters)
label_length = 4
width,height=128,64

# [classes, background]
def encode(chars):
    for_ret = torch.zeros(len(characters))
    for c in chars:
        for_ret[char_pos[c]] = 1
    return for_ret


def decode(tensor):
    """
        向量转字符
    """
    pos = torch.argmax(tensor)
    return characters[pos]


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size):
        super(FakeDataset, self).__init__()
        self.size = dataset_size
        self.width = 128
        self.height = 64
        self.label_length = label_length
        self.n_class = num_classes
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """
        image.shape=[channel,width,height]
        target = [label_length,classes]
        radom_str= [...]
        """
        random_str = "".join(
            np.random.choice(characters, self.label_length, replace=False)
        )
        image = to_tensor(self.generator.generate_image(random_str))
        target = [char_pos[r] for r in random_str]
        return image, torch.tensor(target), random_str

def net_output_right(input_data:torch.tensor,target:torch.tensor) -> float :
    input_data = input_data.detach().cpu()
    input_label = torch.argmax(input_data,dim=2).permute(1,0)
    return (input_label == target.cpu()).all().item()
