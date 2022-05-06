from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T

from PIL import Image, ImageFilter
import torchvision
from torchvision.utils import save_image

class OmniglotReactionTimeDataset(Dataset):
    """
    Dataset for omniglot + reaction time data

    Dasaset Structure:
    label1, label2, real_file, generated_file, reaction time
    ...

    args:
    - path: string - path to dataset (should be a csv file)
    - transforms: torchvision.transforms - transforms on the data
    """

    def __init__(self, data_file, transforms=None):
        self.raw_data = pd.read_csv(data_file)
        self.transform = transforms

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        label1 = int(self.raw_data.iloc[idx, 0])
        label2 = int(self.raw_data.iloc[idx, 1])

        im1name = self.raw_data.iloc[idx, 2]
        image1 = Image.open(im1name)
        im2name = self.raw_data.iloc[idx, 3]
        image2 = Image.open(im2name)
        
        rt = self.raw_data.iloc[idx, 4]
        sigma_or_accuracy = self.raw_data.iloc[idx, 5]
        
        # if you wanted to, you could perturb one of the images. 
        # our final experiments did not do this, though. only some of them 
        # image1 = image1.filter(ImageFilter.GaussianBlur(radius = sigma_or_accuracy))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {'label1': label1, 'label2': label2, 'image1': image1,
                                            'image2': image2, 'rt': rt, 'acc': sigma_or_accuracy}

        return sample

# you can add other dataset classes below

class DataModule(pl.LightningDataModule):

    #TODO just change the dataset name as params and stuff
    # so that the model knows what to load and everything for the script and such
    def __init__(self, data_dir: str = "tiny-imagenet-200", batch_size=64):
        super().__init__()
        self.data_dir = data_dir

        # DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]
        self.TRAIN_DIR = os.path.join(data_dir, 'train')
        self.VALID_DIR = os.path.join(data_dir, 'val')
        self.val_img_dir = os.path.join(self.VALID_DIR, 'images')
        self.batch_size = batch_size


        self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
            ])

    def prepare_data(self):
        # print('self.data_dir ', self.data_dir)
        if self.data_dir == 'tiny-imagenet-200':

            # Open and read val annotations text file
            fp = open(os.path.join(self.VALID_DIR, 'val_annotations.txt'), 'r')
            data = fp.readlines()

            # Create dictionary to store img filename (word 0) and corresponding
            # label (word 1) for every line in the txt file (as key value pair)
            val_img_dict = {}
            for line in data:
                words = line.split('\t')
                val_img_dict[words[0]] = words[1]
            fp.close()

            for img, folder in val_img_dict.items():
                newpath = (os.path.join(self.val_img_dir, folder))
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                if os.path.exists(os.path.join(self.val_img_dir, img)):
                    os.rename(os.path.join(self.val_img_dir, img), os.path.join(newpath, img))

        else: 
            train_transform = T.Compose([
                            # T.Resize(32, padding=0),
                            T.Grayscale(num_output_channels=3),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            ])
            self.dataset = OmniglotReactionTimeDataset('small_dataset.csv', 
                        transforms=train_transform)

            test_split = .2
            shuffle_dataset = True

            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))

            if shuffle_dataset:
                np.random.seed(2)
                np.random.shuffle(indices)
            self.train_indices, self.val_indices = indices[split:], indices[:split]

            self.train_sampler = SubsetRandomSampler(self.train_indices)
            self.val_sampler = SubsetRandomSampler(self.val_indices)

    # setup
    # def setup(self, stage: Optional[str] = None):

    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit" or stage is None:
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    #     # Assign test dataset for use in dataloader(s)
    #     if stage == "test" or stage is None:
    #         self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    #     if stage == "predict" or stage is None:
    #         self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        if self.data_dir == 'tiny-imagenet-200':
            return self.generate_dataloader(self.TRAIN_DIR, "train",
                                  transform=self.transform)
        else: 
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        if self.data_dir == 'tiny-imagenet-200':
            return self.generate_dataloader(self.val_img_dir, "val",
                                  transform=self.transform)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)

    def generate_dataloader(self, data, name, transform):
        if data is None: 
            return None
        
        # Read image files to pytorch dataset using ImageFolder, a generic data 
        # loader where images are in format root/label/filename
        # See https://pytorch.org/vision/stable/datasets.html
        if transform is None:
            dataset = torchvision.datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = torchvision.datasets.ImageFolder(data, transform=transform)

        
        # Wrap image dataset (defined above) in dataloader 
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=(name=="train"))
        
        return dataloader