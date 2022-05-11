from __future__ import absolute_import

from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback, seed_everything
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
import torchvision
from torchvision import transforms as T

from pytorch_lightning.loggers import WandbLogger

from dataset import OmniglotReactionTimeDataset, DataModule
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

        self.default_loss_fn = nn.CrossEntropyLoss()

        # define model - using argparser or someting like tha
        if self.model_name == 'ViT':
            # also has a pre-trained version on ImageNet
            self.model = torchvision.models.vit_b_32(pretrained=True) 
            # temp stuff to work with omniglot set - refactor into a param dict
        elif self.model_name == 'VGG':
            self.model = torchvision.models.vgg16(pretrained=True)
        elif self.model_name == 'googlenet':
            self.model = torchvision.models.googlenet(pretrained=True)
        elif self.model_name == 'alexnet':
            self.model = torchvision.models.alexnet(pretrained=True)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

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
    parser.add_argument('--dataset_name', type=str, default='timy-imagenet-200',
                        help='dataset file to use. out.csv is the full set')
    parser.add_argument('--log', type=bool, default=False,
                        help='log metrics via WandB')

    args = parser.parse_args()


    # 5 seed runs for trials on this data
    for seed_idx in range(1, 6):
        random_seed = seed_idx ** 3
        seed_everything(random_seed, workers=True)

        metrics_callback = MetricCallback()

        wandb_logger = None
        if args.log:
            logger_name = "{}-{}-{}-tinyimagenet".format(args.model_name, args.dataset_name, random_seed)
            wandb_logger = WandbLogger(name=logger_name, project="psychophysics_model_search_02", log_model="all")

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            num_sanity_val_steps=2,
            # gpus=[0,1,2,3] if torch.cuda.is_available() else None,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=4,
            auto_select_gpus=True,
            callbacks=[metrics_callback],
            logger=wandb_logger
        ) 

        model_ft = Model()
        data_module = DataModule(data_dir=args.dataset_name, batch_size=args.batch_size)

        trainer.fit(model_ft, data_module)

        save_name = "{}seed-{}-{}-tinyimagenet.pth".format(random_seed, args.model_name, args.dataset_name)
        trainer.save_checkpoint(save_name)