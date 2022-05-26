from __future__ import absolute_import

# import timm
from transformers import ViTConfig, ViTModel 
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

from dataset import  DataModule, msd_net_dataset
from psychloss import RtPsychCrossEntropyLoss

#####################################
# constants
# TODO: these might need to be updated overtime ...
known_exit_rt = [3.5720, 4.9740, 7.0156, 11.6010, 27.5720]
unknown_exit_rt = [4.2550, 5.9220, 8.2368, 13.0090, 28.1661]

known_thresholds = [0.0035834426525980234, 0.0035834424197673798,
                    0.0035834426525980234, 0.0035834424197673798, 0.0035834424197673798]
unknown_thresholds = [0.0035834426525980234, 0.0035834424197673798,
                      0.0035834426525980234, 0.0035834424197673798, 0.0035834424197673798]
#                                                       
#####################################


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
            configuration = ViTConfig(return_dict=False) # you can edit model params
            self.model = ViTModel(configuration)
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

        elif self.dataset_name == 'tiny-imagenet-200': 
            inputs, labels = batch
            # TODO: change to psych-imagenet datset from lab

            outputs = self.model(inputs)
            loss = self.default_loss_fn(outputs, labels)

            labels_hat = torch.argmax(outputs, dim=1)
            train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        else:
            # Jin's imagenet-modified dataset

            # this is a setting, but we probably don't need it 

            # if i in rt_indices:
            #     batch = next(rt_iter)

            # elif i in no_rt_indices:
            #     try:
            #         batch = next(no_rt_iter)
            #     except:
            #         continue

            # print('batch is here:', batch)
            input = batch["imgs"]
            rts = batch["rts"]
            target = batch["labels"]
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target).long()

            # print('shape of the input is:', input_var.shape)
            # print('shape of the target is:', target.shape)

            # TODO: The model expects the rts and handles them in a custom way 
            # 1 - handle them with our custom loss instead 
            # 2 - later, try them with the MSDensenet, too

            # output, feature, end_time = self.model(input_var)
            outputs = self.model(input_var)
            print('outputs are', outputs)

            if self.loss_name == 'cross_entropy':
                loss = self.default_loss_fn(outputs, target_var)
            else:
                # psych loss
                loss = RtPsychCrossEntropyLoss(outputs, target_var, rts)
            labels_hat = torch.argmax(outputs, dim=1)
            train_acc = torch.sum(target_var.data == labels_hat).item() / (len(target_var) * 1.0)


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

        elif self.dataset_name == 'tiny-imagenet-200': 
            inputs, labels = batch
            # TODO: change to psych-imagenet datset from lab

            outputs = self.model(inputs)
            loss = self.default_loss_fn(outputs, labels)

            labels_hat = torch.argmax(outputs, dim=1)
            val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        else: 
            # this is a setting, but we probably don't need it 

            # if i in rt_indices:
            #     batch = next(rt_iter)

            # elif i in no_rt_indices:
            #     try:
            #         batch = next(no_rt_iter)
            #     except:
            #         continue

            input = batch["imgs"]
            rts = batch["rts"]
            target = batch["labels"]
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target).long()

            # print('shape of the input is:', input_var.shape)
            # print('shape of the target is:', target.shape)

            # TODO: The model expects the rts and handles them in a custom way 
            # 1 - handle them with our custom loss instead 
            # 2 - later, try them with the MSDensenet, too

            # output, feature, end_time = self.model(input_var)
            
            # see where are files compare here ..

            
            outputs = self.model(input_var)
            outputs = outputs[1]
            print('outputs shape is', outputs.shape)
            print('outputs are', outputs)

            
            if self.loss_name == 'cross_entropy':
                loss = self.default_loss_fn(outputs, target_var)
            else:
                # psych loss
                loss = RtPsychCrossEntropyLoss(outputs, target_var, rts)


            labels_hat = torch.argmax(outputs, dim=1)
            val_acc = torch.sum(target_var.data == labels_hat).item() / (len(target_var) * 1.0)

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
    parser.add_argument('--num_classes', type=int, default=335,
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


    #TODO: just add the seeds again at some point
    path = '/afa/crc.nd.edu/user/j/jdulay/.cache/'
    if os.path.isdir(path):
        os.rmdir(path)
    seed_everything(3, workers=True)

    metrics_callback = MetricCallback()

    wandb_logger = None
    if args.log:
        logger_name = "{}-{}-{}-imagenet".format(args.model_name, args.dataset_name, random_seed)
        wandb_logger = WandbLogger(name=logger_name,project="psychophysics_model_search_08")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        devices=4, 
        accelerator='gpu',            
        strategy='ddp',
        auto_select_gpus=True,
        callbacks=[metrics_callback],
        logger=wandb_logger,
        progress_bar_refresh_rate=0
    ) 

    model_ft = Model()

    data_module = DataModule(data_dir=args.dataset_name, batch_size=args.batch_size)

    trainer.fit(model_ft, data_module)
    if os.path.isdir(path):
        os.rmdir(path)
    save_name = "{}seed-{}-{}-imagenet.pth".format(random_seed, args.model_name, args.dataset_name)
    trainer.save_checkpoint(save_name)
    if os.path.isdir(path):
        os.rmdir(path)
