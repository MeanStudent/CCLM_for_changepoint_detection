from utils.marvl_preproc import marvl_preproc
import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn as nn

from models.model_nlvr import NLVRModel

import utils
from torch.utils.data import DataLoader,WeightedRandomSampler
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_pretrain_cclm import CrossViewLM

from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode
from dataset.randaugment import RandomAugment

from dataset.nlvr_dataset import nlvr_dataset
from PIL import Image
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import build_mlp
from sklearn.metrics import precision_recall_curve,auc
import re
from collections import OrderedDict
from optim import create_optimizer
import random

# useful functions and classes
def my_softmax(nvlr_pred):
    return torch.exp(nvlr_pred)/torch.sum(torch.exp(nvlr_pred))

def show_random_sample(train_dataset):
    random_index = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[random_index]
    fig, ax = plt.subplots(1,2,figsize = (10,5))
    print(sample[0].shape)
    ax[0].imshow(sample[0].transpose(2,0).transpose(1,0))
    ax[1].imshow(sample[1].transpose(2,0).transpose(1,0))
    plt.show()
    print(sample[2])
    print(f'label: {True if sample[3]==1 else False}')
    
def show_a_batch(batch_sample):
    batch_size = len(batch_sample[0])
    for batch_idx in range(batch_size):
        img_start = batch_sample[0][batch_idx]
        img_end = batch_sample[1][batch_idx]
        text = batch_sample[2][batch_idx]
        label = batch_sample[3][batch_idx]
        fig, ax = plt.subplots(1,2,figsize = (10,5))
        ax[0].imshow(img_start.transpose(2,0).transpose(1,0))
        ax[1].imshow(img_end.transpose(2,0).transpose(1,0))
        plt.show()
        print(text)
        print(f'label: {label}')
        
class LDCDataset(Dataset):  # dataset used for training.
    def __init__(self, csv_file, transform):
        self.data_csv = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        # get img
        img_start_path = self.data_csv.img_start[idx]
        img_end_path = self.data_csv.img_end[idx]
        img_start = Image.open(img_start_path).convert('RGB')
        img_end = Image.open(img_end_path).convert('RGB')
        # get text
        text = self.data_csv.text[idx]
        # get label
        label = self.data_csv.label[idx]
        
        if self.transform is not None:
            img_start = self.transform(img_start)
            img_end = self.transform(img_end)

        return img_start, img_end, text, label
    
class LDCDataset_val(Dataset):  # dataset used for validation.
    def __init__(self, csv_file, transform):
        self.data_csv = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        # get img
        img_start_path = self.data_csv.img_start[idx]
        img_end_path = self.data_csv.img_end[idx]
        img_start = Image.open(img_start_path).convert('RGB')
        img_end = Image.open(img_end_path).convert('RGB')
        # get text
        text = self.data_csv.text[idx]
        # get label
        label = self.data_csv.label[idx]
        
        if self.transform is not None:
            img_start = self.transform(img_start)
            img_end = self.transform(img_end)

        return img_start, img_end, text, label, self.data_csv.iloc[idx,:]
    
def is_a_changepoint(changepoint_list, time_start, time_end):
    for changepoint in changepoint_list:
        if time_start <= changepoint <= time_end:
            return 1
    return 0


def construct_dataset_csv(dataset,transcribe_name,part = None, use_context = True):
    column_file_id = []
    column_start_time = []
    column_end_time = []
    column_start_frame = []
    column_end_frame = []
    column_text = []
    column_label = []
    file_ids = list(dataset.keys())
    pass_file_num = 0

    if not use_context:
        for file_id in file_ids:
            file_root_path = dataset[file_id]['processed_dir']
            changepoint_list = [changepoint_dict['timestamp'] for changepoint_dict in dataset[file_id]['changepoints']]
            for idx in range(len(dataset[file_id]['utterances'][transcribe_name])):
                sample = dataset[file_id]['utterances'][transcribe_name][idx]
                if len(sample['video_frames'])>=2:
                    column_file_id.append(file_id)
                    column_start_time.append(sample['start'])
                    column_end_time.append(sample['end'])
                    start_frame_path = os.path.join(file_root_path,sample['video_frames'][0][1])
                    end_frame_path = os.path.join(file_root_path,sample['video_frames'][-1][1])
                    column_start_frame.append(start_frame_path)
                    column_end_frame.append(end_frame_path)
                    if len(sample['text'])>512:
                        column_text.append(sample['text'][:500])
                    else:
                        column_text.append(sample['text'])
                    label = is_a_changepoint(changepoint_list, sample['start'],sample['end'])
                    column_label.append(label)
    else:
        for file_id in file_ids:
            file_root_path = dataset[file_id]['processed_dir']
            changepoint_list = [changepoint_dict['timestamp'] for changepoint_dict in dataset[file_id]['changepoints']]
            total_text = []
            for idx in range(len(dataset[file_id]['utterances'][transcribe_name])):
                sample = dataset[file_id]['utterances'][transcribe_name][idx]
                if len(sample['text'])>512:
                        total_text.append(sample['text'][:300])
                else:
                    total_text.append(sample['text'])
            for idx in range(len(dataset[file_id]['utterances'][transcribe_name])):
                sample = dataset[file_id]['utterances'][transcribe_name][idx]
                total_text
                if len(sample['video_frames'])>=2:
                    column_file_id.append(file_id)
                    column_start_time.append(sample['start'])
                    column_end_time.append(sample['end'])
                    start_frame_path = os.path.join(file_root_path,sample['video_frames'][0][1])
                    end_frame_path = os.path.join(file_root_path,sample['video_frames'][-1][1])
                    column_start_frame.append(start_frame_path)
                    column_end_frame.append(end_frame_path)
                    if idx>=10 and idx<=len(dataset[file_id]['utterances'][transcribe_name])-10:
                        pre_context = ' '.join(total_text[idx-10:idx])
                        cur_context = total_text[idx][:256] # cur context only previous 256 tokens
                        post_context = ' '.join(total_text[idx+1:idx+11])
                        left_token_len = 400 - len(cur_context)
                        pre_context = pre_context[max(0,(len(pre_context)-int(left_token_len/2))):]
                        post_context = post_context[:int(left_token_len/2)]
                        final_text = pre_context+'<Pre_Context>'+cur_context+'<Post_Context>'+post_context
                        column_text.append(final_text)
                    elif idx<10:
                        pre_context = ' '.join(total_text[:idx])
                        cur_context = total_text[idx][:256]
                        post_context = ' '.join(total_text[idx+1:idx+11])
                        left_token_len = 400 - len(cur_context)
                        pre_context = pre_context[max(0,(len(pre_context)-int(left_token_len/2))):]
                        post_context = post_context[:int(left_token_len/2)]
                        final_text = pre_context+'<Pre_Context>'+cur_context+'<Post_Context>'+post_context
                        column_text.append(final_text)
                    elif idx>len(dataset[file_id]['utterances'][transcribe_name])-10:
                        pre_context = ' '.join(total_text[idx-10:idx])
                        cur_context = total_text[idx][:256]
                        post_context = ' '.join(total_text[idx+1:])
                        left_token_len = 400 - len(cur_context)

                        pre_context = pre_context[max(0,(len(pre_context)-int(left_token_len/2))):]
                        post_context = post_context[:int(left_token_len/2)]
                        final_text = pre_context+'<Pre_Context>'+cur_context+'<Post_Context>'+post_context
                        column_text.append(final_text)

                    label = is_a_changepoint(changepoint_list, sample['start'],sample['end'])
                    column_label.append(label)

    df = pd.DataFrame({
        'file_id':column_file_id,
        'img_start':column_start_frame,
        'img_end':column_end_frame,
        'text':column_text,
        'label':column_label,
        'time_start':column_start_time,
        'time_end':column_end_time
    })
    if part == 1:
        print(f'total length of dataset:{len(df)}, now inferencing on first half lengts is{int(len(df)/2)}')
        df = df.iloc[:int(len(df)/2),:]
    elif part == 2:
        print(f'total length of dataset:{len(df)}, now inferencing on second half lengts is{int(len(df)/2)}')
        df = df.iloc[int(len(df)/2):,:]
    elif part is None:
        pass
    return df

def create_down_sample_dataloader(train_csv, down_sample_seed, train_batch_size, train_transform,neg_pos_rate,evaluation = True): # down sample dataloader for training and finetune
    # down_sample_csv
    positive_samples = train_csv[train_csv['label'] == 1]
    negative_samples = train_csv[train_csv['label'] == 0]

    sampled_negative_samples = negative_samples.sample(n=int(neg_pos_rate*len(positive_samples)), random_state=down_sample_seed)

    balanced_df = pd.concat([positive_samples, sampled_negative_samples])

    balanced_df = balanced_df.sample(frac=1, random_state=down_sample_seed).reset_index(drop=True)
    
    # create dataset
    world_size = utils.get_world_size()
    if evaluation:
        train_dataset = LDCDataset_val(balanced_df, train_transform)

        data_loader = DataLoader(train_dataset, num_workers=4,batch_size=train_batch_size,shuffle=True,collate_fn = custom_collate_fn)
    else: 
        train_dataset = LDCDataset(balanced_df, train_transform)

        data_loader = DataLoader(train_dataset,num_workers=4, batch_size=train_batch_size,shuffle=True)
    
    return data_loader

# useful functions and classes
def my_softmax(nvlr_pred):
    return torch.exp(nvlr_pred)/torch.sum(torch.exp(nvlr_pred))

def custom_collate_fn(batch):
    image0_batch = torch.stack([item[0] for item in batch])
    image1_batch = torch.stack([item[1] for item in batch])
    texts = [item[2] for item in batch]
    targets_batch = torch.tensor([item[3] for item in batch])
    data_pair_infos = [item[4] for item in batch]

    return image0_batch, image1_batch, texts, targets_batch, data_pair_infos

def calculate_matrix(csv):
    correct_pred = csv[csv['true_labels'] == csv['pred_labels']]
    wrong_pred = csv[csv['true_labels'] != csv['pred_labels']]
    true_positive = correct_pred[correct_pred['true_labels'] == 1]
    false_positive = wrong_pred[wrong_pred['true_labels'] == 0]
    true_negative = correct_pred[correct_pred['true_labels'] == 0]
    false_negative = wrong_pred[wrong_pred['true_labels'] == 1]
    
    precision = len(true_positive)/(len(false_positive)+len(true_positive)+0.000001)
    recall = len(true_positive)/(len(true_positive)+len(false_negative)+0.000001)
    accuracy = len(correct_pred)/(len(correct_pred)+len(wrong_pred))
    precision_auc, recall_auc, thresholds = precision_recall_curve(list(csv.true_labels),list(csv.logi_positive))
    pr_auc = auc(recall_auc, precision_auc)
    print(f'confusion matrix:\n{[len(true_positive), len(false_negative)]}\n{[len(false_positive), len(true_negative)]}')
    print('-----------------------------------------')
    print(f'recall is: {recall}, precision is: {precision}')
    print(f'Accuracy is {accuracy}')

    return recall, precision, accuracy, pr_auc, list(precision_auc), list(recall_auc)
    
def save_top_models(model, auc, top_models, save_dir):
    model_name = f"temp_model_{auc:.4f}.pth"
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path)

    top_models.append((model_name, auc))
    top_models.sort(key=lambda x: x[1], reverse=True)

    if len(top_models) > 3:
        _, removed_auc = top_models.pop()
        os.remove(os.path.join(save_dir, f"temp_model_{removed_auc:.4f}.pth"))

    for i, (filename, auc) in enumerate(top_models):
        temp_model_path = os.path.join(save_dir, filename)
        final_model_path = os.path.join(save_dir, f'Final_fine_tuned_{i + 1}.pth')
        shutil.copyfile(temp_model_path, final_model_path)
        
def eval_on_dataset(model,data_loader,device,tokenizer,save_dir,do_save = False):
    results = []
    clock_start = time.time()
    with torch.no_grad():  
        for cur_idx, batch in enumerate(data_loader):
            images = torch.cat([batch[0], batch[1]], dim=0)
            targets = batch[3]
            images, targets = images.to(device), targets.to(device)   

            text_inputs = tokenizer(batch[2], padding='longest', return_tensors="pt").to(device)  
            try:
                predictions = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)
                predictions = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)
                pred_logis = nn.functional.softmax(predictions, dim=1)
                predicted_classes = torch.argmax(pred_logis, dim=1)

                for batch_idx in range(len(predictions)):
                    data_pair_info = batch[4][batch_idx]
                    file_id = data_pair_info.file_id
                    img_start_path = data_pair_info.img_start
                    start_stamp = data_pair_info.time_start
                    img_end_path = data_pair_info.img_end
                    end_stamp = data_pair_info.time_end
                    results.append({
                    'file_ids': file_id,
                    'start_time_stamps': start_stamp,
                    'end_time_stamps': end_stamp,
                    'start_img_paths': img_start_path,
                    'img_end_paths': img_end_path,
                    'true_labels': targets[batch_idx].item(),
                    'pred_labels': predicted_classes[batch_idx].item(),
                    'logi_positive': pred_logis[batch_idx][1].item(),
                    'logi_negative': pred_logis[batch_idx][0].item(),
                    'texts': batch[2][batch_idx]
                    })
                if do_save:
                    pd.DataFrame(results).to_csv(save_dir)
                else:
                    pass
                clock_end = time.time()
                clock_time_used = round(round(clock_end - clock_start,4)/(cur_idx+1),4)
                print(f'current batch: {cur_idx}, total_batch: {len(data_loader)}, average_eval_time_per_batch: {clock_time_used}',end='\r')
            except KeyError as e:
                print(f"KeyError occurred for a sample: {e}\n batch idx:{cur_idx}")
    return pd.DataFrame(results)

def add_module_to_state_dict(state_dict): #used if pretrained checkpoint are trained without nn.DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = "module." + key  # Add "module." prefix
        new_state_dict[new_key] = value
    return new_state_dict

def remove_module_from_state_dict(state_dict): 
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]  # Remove "module." prefix
        else:
            new_key = key  # Keep the key unchanged if it doesn't start with "module."
        new_state_dict[new_key] = value
    return new_state_dict


def vis_train_epoch(train_result_df):
    fig, ax = plt.subplots(1,4, figsize = (15,5))
    epoch_idx = range(len(train_result_df))

    ax[0].plot(epoch_idx,train_result_df.train_accuracys)
    ax[0].set_title('accuracy')

    ax[1].plot(epoch_idx,train_result_df.train_recalls)
    ax[1].set_title('recall')

    ax[2].plot(epoch_idx,train_result_df.train_precisions)
    ax[2].set_title('precision')

    ax[3].plot(epoch_idx,train_result_df.train_aucs)
    ax[3].set_title('auc')


    ax[0].plot(epoch_idx,train_result_df.val_accuracys)
    ax[1].plot(epoch_idx,train_result_df.val_recalls)
    ax[2].plot(epoch_idx,train_result_df.val_precisions)
    ax[3].plot(epoch_idx,train_result_df.val_aucs)
    plt.show()