import os
from torch.utils.data import Dataset
from PIL import Image

# root_dir = os.path.dirname(os.path.abspath(__file__))

class BatchDataset(Dataset):
    def __init__(self, mode, transform=None):
        """
        mode: 'train' 或 'eval'
        transform: torchvision.transforms 变换
        """
        self.transform = transform
        self.mode = mode
        # 假设有一个 txt 文件存储图片路径和标签
        self.samples = []
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if mode == 'train':
            with open(os.path.join(root_dir, './annotations/train.txt'), 'r') as f:
                for line in f.readlines():
                    img_path, age = line.strip().split()
                    img_path = os.path.join(root_dir, './face_trainset', img_path)
                    self.samples.append((img_path, int(age)))
        elif mode == 'eval':
            with open(os.path.join(root_dir, './annotations/val.txt'), 'r') as f:
                for line in f.readlines():
                    img_path, age = line.strip().split()
                    img_path = os.path.join(root_dir, './valset', img_path)
                    self.samples.append((img_path, int(age)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        filename = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, age, filename