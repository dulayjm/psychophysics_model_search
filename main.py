from __future__ import absolute_import

from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import sample, random
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset

from pytorch_lightning.loggers import WandbLogger

from dataset import OmniglotReactionTimeDataset

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
        self.dataset = args.dataset_file
        self.model_name = args.model_name
        self.loss_name = args.loss_fn
                
        if self.loss_name == 'cross_entropy':
            self.loss_fn=nn.CrossEntropyLoss()
       
        # define model - using argparser or someting like tha
        if self.model_name == 'resnet':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif self.model_name == 'ViT':
            # also has a pre-trained version on ImageNet
            self.model = torchvision.models.vision_transformer(pretrained=True)

    def prepare_data(self):
        train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
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
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):        
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
        psych_tensor = psych_tensor.to(self.device)

        outputs = self.model(inputs).to(self.device)

        if self.loss_name == 'cross_entropy':
            loss = self.loss_fn(outputs, labels)
        elif self.loss_name == 'psych_acc': 
            loss = self.loss_fn(outputs, labels, psych_tensor)
        else:
            loss = self.loss_fn(outputs, labels, psych_tensor)

        # calculate accuracy per class
        labels_hat = torch.argmax(outputs, dim=1)
        train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        print('train loss is', loss)
        print('train acc is ', train_acc)

        return {
            'loss': loss,
            'train_acc': train_acc
        }
    
    def validation_step(self, batch, batch_idx):
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
        psych_tensor = psych_tensor.to(self.device)


        outputs = self.model(inputs).to(self.device)

        if self.loss_name == 'cross_entropy':
            loss = self.loss_fn(outputs, labels)
        elif self.loss_name == 'psych_acc': 
            loss = self.loss_fn(outputs, labels, psych_tensor)
        else:
            loss = self.loss_fn(outputs, labels, psych_tensor)

        # calculate accuracy per class
        labels_hat = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        print('val loss is', loss)
        print('val acc is ', val_acc)

        return {
            'val_loss': loss,
            'val_acc': val_acc
        }
    
if __name__ == '__main__':
    # TODO: add working logger

    # args
    parser = ArgumentParser(description='Training Psych Loss.')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='number of epochs to use')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.015, 
                        help='learning rate')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy',
                        help='loss function to use. select: cross_entropy-entropy, psych_rt, psych_acc')
    parser.add_argument('--model_name', type=str, default='resnet',
                        help='model architecfture to use.')                
    parser.add_argument('--dataset_file', type=str, default='small_dataset.csv',
                        help='dataset file to use. out.csv is the full set')
    parser.add_argument('--log', type=bool, default=False,
                        help='log metrics via neptune')
    
    args = parser.parse_args()

    metrics_callback = MetricCallback()
    # wandb_logger = WandbLogger(project="psychophysics_model_search", log_model="all")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        num_sanity_val_steps=2,
        gpus=[3] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        # logger=wandb_logger
    ) 

    model_ft = Model()

    trainer.fit(model_ft)