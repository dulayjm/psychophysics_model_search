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

from timeit import default_timer as timer

# *** DATASET is located at ***
# /scratch365/jhuang24/dataset_v1_3_partition/train_valid

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
    def __init__(self, data_dir: str = "tiny-imagenet-200", batch_size=64):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size
        # DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]
        if data_dir == 'tiny-imagenet-200':
            self.TRAIN_DIR = os.path.join(data_dir, 'train')
            self.VALID_DIR = os.path.join(data_dir, 'val')
            self.val_img_dir = os.path.join(self.VALID_DIR, 'images')
            self.batch_size = batch_size

            self.transform = T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
                ])

        elif data_dir == 'psych-rt':
            pass

        else: 
            # save_path_base = "/afs/crc.nd.edu/user/j/jdulay/research/psychophysics_model_search"

            # use_performance_loss = False
            # use_exit_loss = True

            cross_entropy_weight = 1.0
            perform_loss_weight = 1.0
            exit_loss_weight = 1.0

            # save_path_sub = "known_only_cross_entropy_" + str(cross_entropy_weight) + \
            #                "_pfm_" + str(perform_loss_weight)


            json_data_base = '/afs/crc.nd.edu/user/j/jdulay'


            #/print('DEBUG the path is here', json_data_base)


            use_json_data = True
            save_training_prob = False

            # cherry-picked
            self.train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
            self.valid_known_known_with_rt_path = os.path.join(json_data_base, "valid_known_known_with_rt.json")
            # self.test_known_known_with_rt_path = os.path.join(json_data_base, "test_known_known_with_rt.json")


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

        elif self.data_dir == 'psych-rt': 
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

        else:
            # transforms here 
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

            # and this one hehe
            self.valid_known_known_with_rt_dataset = msd_net_dataset(json_path=self.valid_known_known_with_rt_path,
                                                                    transform=valid_transform) 


    def train_dataloader(self):
        if self.data_dir == 'tiny-imagenet-200':
            return self._generate_dataloader(self.TRAIN_DIR, "train",
                                  transform=self.transform)
        elif self.data_dir == 'psych-rt': 
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        else: 
            return DataLoader(self.train_known_known_with_rt_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                collate_fn=self.collate
                                                )


    def val_dataloader(self):
        if self.data_dir == 'tiny-imagenet-200':
            return self._generate_dataloader(self.val_img_dir, "val",
                                  transform=self.transform)
        elif self.data_dir == 'psych-rt': 
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)
        else: 
            return DataLoader(self.valid_known_known_with_rt_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                collate_fn=self.collate
                                                )
    # helper function for tiny class
    def _generate_dataloader(self, data, name, transform):
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

        return {
            "img": img,
            "gt_label": item["label"],
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
