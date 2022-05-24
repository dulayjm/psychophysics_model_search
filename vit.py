from __future__ import absolute_import

# import timm
from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback, seed_everything
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import json

from pytorch_lightning.loggers import WandbLogger

from psychloss import RtPsychCrossEntropyLoss

from transformers import ViTForImageClassification, AdamW
import torch.nn as nn

from PIL import Image
from transformers import ViTFeatureExtractor

from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from psychloss import RtPsychCrossEntropyLoss

# setup the data _ad-hoc_ first, for the imagenet style stuff 
class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def collate_fn(examples):
    pixel_values = torch.stack([example["img"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    rt = torch.tensor([example["rt"] for example in examples])

    return {"img": pixel_values, "label": labels, "rt": rt}


class CustomDataModule(pl.LightningDataModule):
    def __init__(self):
        batch_size = 16
        self.num_labels = 1000
        json_data_base = '/afs/crc.nd.edu/user/j/jdulay'

        self.train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
        self.valid_known_known_with_rt_path = os.path.join(json_data_base, "valid_known_known_with_rt.json")

        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    def prepare_data(self):
        normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        train_transforms = Compose(
                [
                    RandomResizedCrop(self.feature_extractor.size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )

        val_transforms = Compose(
                [
                    Resize(self.feature_extractor.size),
                    CenterCrop(self.feature_extractor.size),
                    ToTensor(),
                    normalize,
                ]
            )

        self.train_known_known_with_rt_dataset = msd_net_dataset(json_path=self.train_known_known_with_rt_path,
                                                                transform=train_transforms)

        # and this one hehe
        self.valid_known_known_with_rt_dataset = msd_net_dataset(json_path=self.valid_known_known_with_rt_path,
                                                                transform=val_transforms) 

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_known_known_with_rt_dataset, batch_size=16, collate_fn=collate_fn)
        return train_dataloader
        
    def val_dataloader(self):
        val_dataloader = DataLoader(self.valid_known_known_with_rt_dataset, batch_size=16, collate_fn=collate_fn)
        return val_dataloader


class msd_net_dataset(Dataset):
    def __init__(self,
                 json_path,
                 transform):

        with open(json_path) as f:
            data = json.load(f)
        #print("Json file loaded: %s" % json_path)

        self.data = data
        self.transform = transform
        self.random_weight = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[str(idx)]

        # Open the image and do normalization and augmentation
        img = Image.open(item["img_path"])
        img = img.convert('RGB')
        # print(item["img_path"])
        # print(item["label"])
        # print(img.size) 
        # print('type of the image before transform: ', type(img))
        img = self.transform(img)

        # Deal with reaction times
        if self.random_weight is None:
            # print("Checking whether an RT exists for this image...")
            if item["RT"] != None:
                rt = item["RT"]
            else:
                # print("RT does not exist")
                rt = None
        # No random weights for reaction time
        else:
            pass

        return {
            "img": img,
            "label": item["label"],
            "rt": rt,
            "category": item["category"]
        }

class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1000)
        self.num_labels=10000
        self.criterion = nn.CrossEntropyLoss()
        #   num_labels=10,
        #   id2label=id2label,
        #   label2id=label2id

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        # TODO: implement w RT 
        #print('beginning of common step')
        pixel_values = batch['img']
        #print('pixel_values', pixel_values.shape, pixel_values)
        labels = batch['label']
        #print('lbaels,', labels.shape, labels)
        logits = self(pixel_values)
        #print('logits', logits.shape, logits)
        rts = batch['rt']

        if args.loss_fn == 'psych':
            loss = RtPsychCrossEntropyLoss(logits, labels, rts)
        else:
            loss = self.criterion(logits, labels)
            
        predictions = logits.argmax(-1)
        #print('debug here')
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)


if __name__ == '__main__':
    # args
    parser = ArgumentParser(description='Neural Architecture Search for Psychophysics')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='number of epochs to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=1000,
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

    wandb_logger = None
    if args.log:
        logger_name = "{}-{}-{}-imagenet".format(args.model_name, args.dataset_name, 'DEBUG')
        wandb_logger = WandbLogger(name=logger_name, project="ViT-DEBUG", log_model="all")
    
    metrics_callback = MetricCallback()

    # because all of this fits in a data module
    
    datamodule = CustomDataModule()
    model = ViTLightningModule()
    trainer = pl.Trainer(
        max_epochs=20, 
        gpus=-1 if torch.cuda.is_available() else None, 
        auto_select_gpus=True, 
        logger=wandb_logger,
        callbacks=[metrics_callback],
        progress_bar_refresh_rate=0
    )

    trainer.fit(model, datamodule)

    save_name = "{}seed-{}-{}-imagenet.pth".format('DEBUG', args.model_name, args.dataset_name)
    trainer.save_checkpoint(save_name)
