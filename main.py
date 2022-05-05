from __future__ import absolute_import

from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T

from pytorch_lightning.loggers import WandbLogger

from dataset import OmniglotReactionTimeDataset
from psychloss import RtPsychCrossEntropyLoss

class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class Model(LightningModule):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = args.learning_rate
        self.batch_size= args.batch_size
        self.num_classes = args.num_classes
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.loss_name = args.loss_fn

        # if self.loss_name == 'cross_entropy':
        self.default_loss_fn = nn.CrossEntropyLoss()

        # define model - using argparser or someting like tha
        if self.model_name == 'resnet':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif self.model_name == 'ViT':
            # also has a pre-trained version on ImageNet
            self.model = torchvision.models.vit_b_32(pretrained=True) 
            # temp stuff to work with omniglot set - refactor into a param dict

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

    def prepare_data(self):
        if self.dataset_name == 'imagenet':
            DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

            # Define training and validation data paths
            self.TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
            VALID_DIR = os.path.join(DATA_DIR, 'val')


            self.preprocess_transform_pretrain = T.Compose([
                            T.Resize(256), # Resize images to 256 x 256
                            T.CenterCrop(224), # Center crop image
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),  # Converting cropped images to tensors
                            T.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
            ])
            self.val_img_dir = os.path.join(VALID_DIR, 'images')

            # Open and read val annotations text file
            fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
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

    def train_dataloader(self):
        if self.dataset_name == 'imagenet':
            return self.generate_dataloader(self.TRAIN_DIR, "train",
                                  transform=self.preprocess_transform_pretrain)
        else: 
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        if self.dataset_name == 'imagenet':
            return self.generate_dataloader(self.val_img_dir, "val",
                                  transform=self.preprocess_transform_pretrain)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):  

        if self.dataset_name == 'psych_rt':     
            image1 = batch['image1']
            image2 = batch['image2']

            label1 = batch['label1']
            label2 = batch['label2']

            if self.loss_name == 'psych-acc':
                psych = batch['acc']
            else: 
                psych = batch['rt']

            # concatenate the batched images
            inputs = torch.cat([image1, image2], dim=0)
            labels = torch.cat([label1, label2], dim=0)

            # apply psychophysical annotations to correct images
            psych_tensor = torch.zeros(len(labels))
            j = 0 
            for i in range(len(psych_tensor)):
                if i % 2 == 0: 
                    psych_tensor[i] = psych[j]
                    j += 1
                else: 
                    psych_tensor[i] = psych_tensor[i-1]
            psych_tensor = psych_tensor

            outputs = self.model(inputs)

            loss = None
            if self.loss_name == 'psych_rt':
                loss = RtPsychCrossEntropyLoss(outputs, labels, psych_tensor)
            else:
                loss = self.default_loss_fn(outputs, labels)
            # calculate accuracy per class
            labels_hat = torch.argmax(outputs, dim=1)
            train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)


        else: 
            inputs, labels = batch
            # TODO: change to psych-imagenet datset from lab

            outputs = self.model(inputs)
            loss = self.default_loss_fn(outputs, labels)


            labels_hat = torch.argmax(outputs, dim=1)
            train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)


        print('train loss is', loss)
        print('train acc is ', train_acc)
        
        self.log('train_loss', loss)
        self.log('train_acc', train_acc)

        return {
            'loss': loss,
            'train_acc': train_acc
        }
    
    def validation_step(self, batch, batch_idx):
        
        if self.dataset_name == "psych_rt":
            image1 = batch['image1']
            image2 = batch['image2']

            label1 = batch['label1']
            label2 = batch['label2']

            if self.loss_name == 'psych-acc':
                psych = batch['acc']
            else: 
                psych = batch['rt']

            # concatenate the batched images
            inputs = torch.cat([image1, image2], dim=0)
            labels = torch.cat([label1, label2], dim=0)

            # apply psychophysical annotations to correct images
            psych_tensor = torch.zeros(len(labels))
            j = 0 
            for i in range(len(psych_tensor)):
                if i % 2 == 0: 
                    psych_tensor[i] = psych[j]
                    j += 1
                else: 
                    psych_tensor[i] = psych_tensor[i-1]
            psych_tensor = psych_tensor


            outputs = self.model(inputs)

            loss = None
            if self.loss_name == 'psych_rt':
                loss = RtPsychCrossEntropyLoss(outputs, labels, psych_tensor)
            else:
                print('sanity here')
                loss = self.default_loss_fn(outputs, labels)

            # calculate accuracy per class
            labels_hat = torch.argmax(outputs, dim=1)
            val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        else: 
            inputs, labels = batch
            # TODO: change to psych-imagenet datset from lab

            outputs = self.model(inputs)
            loss = self.default_loss_fn(outputs, labels)

            labels_hat = torch.argmax(outputs, dim=1)
            val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        print('val loss is', loss)
        print('val acc is ', val_acc)

        self.log('val_loss', loss)
        self.log('val_acc', val_acc)

        return {
            'val_loss': loss,
            'val_acc': val_acc
        }
    
if __name__ == '__main__':
    # args
    parser = ArgumentParser(description='Neural Architecture Search for Psychophysics')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='number of epochs to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.015, 
                        help='learning rate')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy',
                        help='loss function to use. select: cross_entropy-entropy, psych_rt, psych_acc')
    parser.add_argument('--model_name', type=str, default='resnet',
                        help='model architecfture to use.')                
    parser.add_argument('--dataset_name', type=str, default='imagenet',
                        help='dataset file to use. out.csv is the full set')
    parser.add_argument('--log', type=bool, default=False,
                        help='log metrics via neptune')
    
    args = parser.parse_args()

    metrics_callback = MetricCallback()
    wandb_logger = WandbLogger(name='sandbox_rt', project="psychophysics_model_search", log_model="all")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        num_sanity_val_steps=2,
        gpus=[3] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        logger=wandb_logger
    ) 

    model_ft = Model()

    trainer.fit(model_ft)