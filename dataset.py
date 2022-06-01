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

import json

import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
from PIL import Image
import sys
import random

import math
# from tqdm import tqdm
from utils.idx_to_class import get_class_to_idx

from timeit import default_timer as timer

idx_to_class = get_class_to_idx()

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        cross_entropy_weight = 1.0
        perform_loss_weight = 1.0
        exit_loss_weight = 1.0
        json_data_base = '/afs/crc.nd.edu/user/j/jdulay'

        self.train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
        self.valid_known_known_with_rt_path = os.path.join(json_data_base, "valid_known_known_with_rt.json")


        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        train_transform = T.Compose([T.Resize((224,224), interpolation=3),
                                                T.ToTensor(),
                                                normalize]) 
        valid_transform = train_transform

        test_transform = T.Compose([T.CenterCrop(224),
                                                T.ToTensor(),
                                                normalize])


        self.train_known_known_with_rt_dataset = msd_net_dataset(json_path=self.train_known_known_with_rt_path,
                                                                    transform=train_transform)

        self.valid_known_known_with_rt_dataset = msd_net_dataset(json_path=self.valid_known_known_with_rt_path,
                                                                    transform=valid_transform) 


    def train_dataloader(self):
        return DataLoader(self.train_known_known_with_rt_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                drop_last=False,
                                                collate_fn=self.collate
                                                )


    def val_dataloader(self):
        return DataLoader(self.valid_known_known_with_rt_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                drop_last=False,
                                                collate_fn=self.collate
                                                )


    def test_dataloader(self): # why is this not registreing ...
        return DataLoader(self.valid_known_known_with_rt_dataset,
                                                batch_size=16,
                                                shuffle=True,
                                                drop_last=False,
                                                collate_fn=self.collate
                                                )

    def collate(self, batch):
        try:
            PADDING_CONSTANT = 0

            batch = [b for b in batch if b is not None]

            #These all should be the same size or error
            #assert len(set([b["img"].shape[0] for b in batch])) == 1
            #assert len(set([b["img"].shape[2] for b in batch])) == 1

            # TODO: what is dim 0, 1, 2??
            """
            dim0: channel
            dim1: ??
            dim2: hight?
            """
            dim0 = batch[0]["img"].shape[0]
            # print('dim0 is :', dim0)
            dim1 = max([b["img"].shape[1] for b in batch])
            dim1 = dim1 + (dim0 - (dim1 % dim0))
            dim1 = 224 # hardcoding
            # print('dim1 is :', dim1)
            dim2 = batch[0]["img"].shape[2]
            dim2 = 224 # hardcoding
            # print('dim2 is :', dim2)

            # print(batch)

            all_labels = []
            psychs = []

            input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
            # input_batch = batch

            for i in range(len(batch)):
                # labels in collate should already be adjusted from __getitem__
                b_img = batch[i]["img"]
                input_batch[i,:,:b_img.shape[1],:] = b_img
                l = batch[i]["gt_label"]
                psych = batch[i]["rt"]
                cate = batch[i]["category"]
                all_labels.append(l)

                # TODO: Leave the scale factor alone for now
                if psych is not None:
                    psychs.append(psych)
                else:
                    psychs.append(0)

            line_imgs = torch.from_numpy(input_batch)
            labels = torch.from_numpy(np.array(all_labels).astype(np.int32))

            return {"imgs": line_imgs,
                    "labels": labels,
                    "rts": psychs,
                    "category": cate}

        except Exception as e:
            print(e)


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

        # should prevent ViT breaking 
        # while using precise number of classes
        orig_label = item['label']
        re_index_label = idx_to_class[orig_label]

        return {
            "img": img,
            "gt_label": re_index_label, 
            "rt": rt,
            "category": item["category"]
        }

def collate_new(batch):
    return batch


# class msd_net_with_grouped_rts(Dataset):
#     def __init__(self,
#                  json_path,
#                  transform,
#                  nb_samples=16,
#                  img_height=32,
#                  augmentation=False):

#         with open(json_path) as f:
#             data = json.load(f)
#         print("Json file loaded: %s" % json_path)

#         self.img_height = img_height
#         self.nb_samples = nb_samples
#         self.data = data
#         self.transform = transform
#         self.augmentation = augmentation
#         self.random_weight = None

#     # TODO: What does this do and how does it influence the training?
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         try:
#             item = self.data[str(idx)]
#         except KeyError:
#             item = self.data[str(idx+1)]

#         # There should be 16 samples in each big batch
#         PADDING_CONSTANT = 0
#         assert(len(item)==self.nb_samples)

#         batch = []

#         # TODO: Process each sample in big batch
#         for i in range(len(item)):
#             one_sample_dict = item[str(i)]

#             # Load the image and do transform
#             img = Image.open(one_sample_dict["img_path"])
#             img = img.convert('RGB')

#             try:
#                 img = self.transform(img)
#             except Exception as e:
#                 print(e)
#                 print(idx)
#                 print(self.data[str(idx)])
#                 sys.exit(0)

#             # Deal with reaction times
#             if self.random_weight is None:
#                 if one_sample_dict["RT"] != None:
#                     rt = one_sample_dict["RT"]
#                 else:
#                     rt = None
#             else:
#                 pass

#             # Append one dictionary to batch
#             batch.append({"img": img,
#                           "gt_label": one_sample_dict["label"],
#                           "rt": rt,
#                           "category": one_sample_dict["category"]})

#         # Put the process that was originally in collate function here
#         assert len(set([b["img"].shape[0] for b in batch])) == 1
#         assert len(set([b["img"].shape[2] for b in batch])) == 1

#         dim_0 = batch[0]["img"].shape[0]
#         dim_1 = max([b["img"].shape[1] for b in batch])
#         dim_1 = dim_1 + (dim_0 - (dim_1 % dim_0))
#         dim_2 = batch[0]["img"].shape[2]

#         all_labels = []
#         psychs = []

#         input_batch = np.full((len(batch), dim_0, dim_1, dim_2), PADDING_CONSTANT).astype(np.float32)

#         for i in range(len(batch)):
#             b_img = batch[i]["img"]
#             input_batch[i, :, :b_img.shape[1], :] = b_img
#             l = batch[i]["gt_label"]
#             psych = batch[i]["rt"]
#             cate = batch[i]["category"]
#             all_labels.append(l)

#             # Check the scale factor alone for now
#             if psych is not None:
#                 psychs.append(psych)
#             else:
#                 psychs.append(0)

#         line_imgs = torch.from_numpy(input_batch)
#         labels = torch.from_numpy(np.array(all_labels).astype(np.int32))

#         print(line_imgs.shape)

#         return {"imgs": line_imgs,
#                 "labels": labels,
#                 "rts": psychs,
#                 "category": cate}




# def save_probs_and_features(test_loader,
#                             model,
#                             test_type,
#                             use_msd_net,
#                             epoch_index,
#                             npy_save_dir,
#                             part_index=None):
#     """
#     batch size is always one for testing.

#     :param test_loader:
#     :param model:
#     :param test_unknown:
#     :param use_msd_net:
#     :return:
#     """

#     # Set the model to evaluation mode
#     model.cuda()
#     model.eval()

#     if use_msd_net:
#         sm = torch.nn.Softmax(dim=2)

#         # For MSD-Net, save everything into npy files
#         full_original_label_list = []
#         full_prob_list = []
#         full_rt_list = []
#         full_feature_list = []


#         for i in tqdm(range(len(test_loader))):
#             try:
#                 batch = next(iter(test_loader))
#             except:
#                 continue

#             input = batch["imgs"]
#             target = batch["labels"]

#             rts = []
#             input = input.cuda()
#             target = target.cuda()

#             # Save original labels to the list
#             original_label_list = np.array(target.cpu().tolist())
#             for label in original_label_list:
#                 full_original_label_list.append(label)

#             input_var = torch.autograd.Variable(input)

#             # Get the model outputs and RTs
#             start = timer()
#             output, feature, end_time = model(input_var)

#             # Handle the features
#             feature = feature[0][0].cpu().detach().numpy()
#             feature = np.reshape(feature, (1, feature.shape[0] * feature.shape[1] * feature.shape[2]))

#             for one_feature in feature.tolist():
#                 full_feature_list.append(one_feature)

#             # Save the RTs
#             for end in end_time[0]:
#                 rts.append(end-start)
#             full_rt_list.append(rts)

#             # extract the probability and apply our threshold
#             prob = sm(torch.stack(output).to()) # Shape is [block, batch, class]
#             prob_list = np.array(prob.cpu().tolist())

#             # Reshape it into [batch, block, class]
#             prob_list = np.reshape(prob_list,
#                                     (prob_list.shape[1],
#                                      prob_list.shape[0],
#                                      prob_list.shape[2]))

#             for one_prob in prob_list.tolist():
#                 full_prob_list.append(one_prob)

#             print(np.asarray(full_feature_list).shape)

#         # Save all results to npy
#         full_original_label_list_np = np.array(full_original_label_list)
#         full_prob_list_np = np.array(full_prob_list)
#         full_rt_list_np = np.array(full_rt_list)
#         full_feature_list_np = np.array(full_feature_list)

#         if part_index is not None:
#             save_prob_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_probs.npy"
#             save_label_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_labels.npy"
#             save_rt_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_rts.npy"
#             save_feature_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_features.npy"

#         else:
#             save_prob_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_probs.npy"
#             save_label_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_labels.npy"
#             save_rt_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_rts.npy"
#             save_feature_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_features.npy"

#         print("Saving probabilities to %s" % save_prob_path)
#         np.save(save_prob_path, full_prob_list_np)
#         print("Saving original labels to %s" % save_label_path)
#         np.save(save_label_path, full_original_label_list_np)
#         print("Saving RTs to %s" % save_rt_path)
#         np.save(save_rt_path, full_rt_list_np)
#         print("Saving features to %s" % save_feature_path)
#         np.save(save_feature_path, full_feature_list_np)
