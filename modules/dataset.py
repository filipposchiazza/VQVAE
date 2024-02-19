import os
import torch
import torch.utils.data as data
from torchvision.io import read_image


class ImageDataset(data.Dataset):

    def __init__(self, 
                 img_dir, 
                 transform=None, 
                 fraction=1.0):
        super(ImageDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.fraction = fraction
        self.img_filenames = os.listdir(img_dir)
        self.img_filenames = self.img_filenames[:int(len(self.img_filenames) * self.fraction)]

    def __len__(self):
        return len(self.img_filenames)
        
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        img = img[:3, :, :]
        img = img / 255.0
        if self.transform:
            img = self.transform(img)
        return img
    


class SubpatchImageDataset(ImageDataset):

    def __init__(self, 
                 img_dir, 
                 img_size, 
                 patch_size,
                 seed=1,
                 transform=None, 
                 fraction=1.0):
        super(SubpatchImageDataset, self).__init__(img_dir, transform, fraction)
        assert img_size > patch_size, "img_size must be greater than patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        img = super().__getitem__(idx)  # (3, img_size, img_size)
        # extract random patch
        x = torch.randint(0, self.img_size - self.patch_size, (1,)).item()
        y = torch.randint(0, self.img_size - self.patch_size, (1,)).item()
        img = img[:, x:x+self.patch_size, y:y+self.patch_size]
        return img



def prepare_ImageDataset(img_dir, 
                         batch_size,
                         validation_split,
                         img_size=256,
                         transform=None,
                         seed=123, 
                         fraction=1.0,
                         subpatch=False,
                         subpatch_size=128):
    if subpatch == False:
        dataset = ImageDataset(img_dir, transform, fraction)
    else:
        dataset = SubpatchImageDataset(img_dir, 
                                       img_size, 
                                       subpatch_size, 
                                       seed, 
                                       transform, 
                                       fraction)    
    val_len = int(len(dataset) * validation_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = data.random_split(dataset, 
                                                   lengths=[train_len, val_len], 
                                                   generator=generator)
    train_dataloader = data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=4)
    val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
    return train_dataset, val_dataset, train_dataloader, val_dataloader
    