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

# ad-hoc dataset additions, for now 
# also, just with tiny-imagenet-for now
# then we can edit everything on the server 

# setup the data _ad-hoc_ first, for the imagenet style stuff 
class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class msd_net_dataset(Dataset):
    def __init__(self,
                 json_path,
                 transform,
                 img_height=32,
                 augmentation=False):

        with open(json_path) as f:
            data = json.load(f)
        #print("Json file loaded: %s" % json_path)

        self.img_height = img_height
        self.data = data
        self.transform = transform
        self.augmentation = augmentation
        self.random_weight = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[str(idx)]
            # print("@" * 20)
            # print(idx)

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
            "gt_label": item["label"],
            "rt": rt,
            "category": item["category"]
        }




class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        #   num_labels=10,
        #   id2label=id2label,
        #   label2id=label2id

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        # TODO: implement w RT 
        pixel_values = batch['img']
        labels = batch['label']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
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

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


if __name__ == '__name__':
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

    print('here0')

    wandb_logger = None
    if args.log:
        logger_name = "{}-{}-{}-imagenet".format(args.model_name, args.dataset_name, 'DEBUG')
        wandb_logger = WandbLogger(name=logger_name, project="ViT-DEBUG", log_model="all")
    
    metrics_callback = MetricCallback()



    batch_size = 16

    json_data_base = '/afs/crc.nd.edu/user/j/jdulay'

    train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
    valid_known_known_with_rt_path = os.path.join(json_data_base, "valid_known_known_with_rt.json")

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


    print('here1')
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )


    train_known_known_with_rt_dataset = msd_net_dataset(json_path=train_known_known_with_rt_path,
                                                            transform=train_transforms)

    # and this one hehe
    valid_known_known_with_rt_dataset = msd_net_dataset(json_path=valid_known_known_with_rt_path,
                                                            transform=val_transforms) 

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     labels = torch.tensor([example["label"] for example in examples])
    #     return {"pixel_values": pixel_values, "labels": labels}

    train_batch_size = 16
    eval_batch_size = 16

    train_dataloader = DataLoader(train_known_known_with_rt_dataset, shuffle=True, batch_size=train_batch_size)
    val_dataloader = DataLoader(valid_known_known_with_rt_dataset, batch_size=eval_batch_size)





    model = ViTLightningModule()
    trainer = pl.Trainer(max_epochs=20, gpus=-1, callbacks=[metrics_callback])
    trainer.fit(model)

    print('here2')
    save_name = "{}seed-{}-{}-imagenet.pth".format('DEBUG', args.model_name, args.dataset_name)
    trainer.save_checkpoint(save_name)