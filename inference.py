import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.marvl_preproc import marvl_preproc
import argparse
import os
import sys
import math
import pandas as pd
from sklearn import metrics

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_nlvr import NLVRModel

import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_pretrain_cclm import CrossViewLM

from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision import transforms

from torchvision.transforms import InterpolationMode
from dataset.randaugment import RandomAugment

from dataset.nlvr_dataset import nlvr_dataset

import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import build_mlp
import re
from collections import OrderedDict
from optim import create_optimizer
from torch.utils.data import Dataset
from torchvision.io import read_image
device = 'cuda'
import random
from sklearn.metrics import precision_recall_curve
import re
import argparse



from useful_functions import *
# remember to set GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--dataset_name", required=True)
    parser.add_argument("-dsp", "--dataset_path", default = '/mnt/swordfish-pool2/ccu/amith-cache.pkl')
    parser.add_argument("-bs", "--batch_size", type = int, required=True)
    parser.add_argument("-uc", "--use_context", type = int, default = 1)
    parser.add_argument("-gpu", "--gpu_index", required=True)
    parser.add_argument("-part", "--part", type=int, default = None)
    parser.add_argument("-msp", "--model_save_path", type=str, default = '/mnt/swordfish-pool2/kh3074/neg_pos_rate2/saved_models/model_tuned_epoch_8')
    parser.add_argument("-rsp", "--result_save_path", type=str, default = '/mnt/swordfish-pool2/kh3074/neg_pos_rate2/evaluate_results/')
    
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    print(f'using {args.gpu_index}')

    def val_dataset_with_model(model_path,dataset, save_dir, test, transcribe_name,args):
        #load model
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        my_nvlr_model_loaded = NLVRModel(config=config)
        if args.use_context:
            print('using resize_token_embeddings')
            special_tokens = ['<Pre_Context>', '<Post_Context>']
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            my_nvlr_model_loaded.text_encoder.resize_token_embeddings(len(tokenizer))
        

        my_nvlr_model_loaded = nn.DataParallel(my_nvlr_model_loaded,device_ids=list(range(len(args.gpu_index.split(',')))))
        checkpoint_ldc = torch.load(model_path, map_location=device)
        #checkpoint_ldc = remove_module_from_state_dict(checkpoint_ldc)
        my_nvlr_model_loaded.load_state_dict(checkpoint_ldc) # load_nlvr_pretrain= False because current checkpoint is CCLM model
        my_nvlr_model_loaded.to(device)
        my_nvlr_model_loaded.eval()




        dataset_csv = construct_dataset_csv(dataset,transcribe_name,part = args.part,use_context = args.use_context).reset_index(drop=True)
        if test:
            dataset_csv = dataset_csv.sample(n=1000, random_state=2333).reset_index(drop=True)
        final_dataset = LDCDataset_val(dataset_csv, val_transform)


        dataloader = DataLoader(final_dataset, batch_size=args.batch_size,shuffle=True,collate_fn = custom_collate_fn,num_workers=4)
        result_df = eval_on_dataset(my_nvlr_model_loaded,dataloader,device, tokenizer,save_dir,do_save = True)
        result_df.to_csv(save_dir)
        return result_df
    # load configs and checkpoints
    #config = yaml.load(open('configs/Pretrain_3m.yaml', 'r'), Loader=yaml.Loader)
    config = yaml.load(open('configs/Pretrain_4m.yaml', 'r'), Loader=yaml.Loader)
    #checkpoint = torch.load('data/cclm_3m_epoch_29.th', map_location=device)
    #checkpoint = torch.load('data/cclm_4m_epoch_29.th', map_location=device) # for training, need to load this

    # dataset transformations
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    train_transform = transforms.Compose([
        transforms.CenterCrop(config['image_res'],# scale=(0.5, 1.0),
                                     #interpolation=InterpolationMode.BICUBIC
                             ),
        #transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'Brightness', 'Sharpness',
                                              'TranslateX', 'TranslateY']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # train test split

    #val_video_num = 50

    #csv_file = pd.read_csv(csv_file_path)
    #
    #file_ids = list(set(list(csv_file.file_id)))
    #random.seed(seed)
    #
    #val_videos = random.sample(file_ids, val_video_num)
    #
    #csv_file['train_test'] = csv_file['file_id'].apply(lambda x: 'val' if x in val_videos else 'train')


    def prepare_refs(val_dataset):
        refs = []
        fileid_list = list(val_dataset.keys())
        for file_id in fileid_list:
            change_points = val_dataset[file_id]['changepoints']
            for change_point in change_points:
                new_changepoint = {'file_id':file_id,
                                  'timestamp':change_point['timestamp'],
                                  'type':val_dataset[file_id]['data_type'],
                                  'impact_scalar':change_point['impact_scalar']}
                refs.append(new_changepoint)
        return refs
    
    def prepare_hyps(eval_df):
        hyps = []
        for row_idx in range(len(eval_df)):
            row = eval_df.iloc[row_idx,:]
            if row['logi_positive'] < row['logi_negative']:
                pass
            else:
                pred_change_point = {'file_id':row['file_ids'],
                                    'timestamp':(row['start_time_stamps']+row['end_time_stamps'])/2,
                                    'type': 'video',
                                    'llr': math.log(row['logi_positive']/row['logi_negative'])}
                hyps.append(pred_change_point)
        return hyps
    sys.path.append('nist_scorer')
    from nist_scorer.CCU_validation_scoring.score_changepoint import score_cp


    def filter_system_preds(system_preds, text_char_threshold, time_sec_threshold, filtering):
        if filtering == 'none':
            return system_preds

        assert filtering in {'highest', 'lowest', 'most_similar'}

        by_file = defaultdict(list)
        for system_pred in system_preds:
            by_file[system_pred['file_id']].append(system_pred)

        filtered_system_preds = []
        for file_id, file_system_preds in by_file.items():
            file_system_preds = list(sorted(
                by_file[file_id],
                key=lambda file_pred: float(file_pred['timestamp'])
            ))

            if len(file_system_preds) <= 1:
                filtered_system_preds.extend(file_system_preds)
                continue

            if file_system_preds[0]['type'] == 'text':
                distance_threshold = text_char_threshold
            else:
                assert file_system_preds[0]['type'] in {'audio', 'video'}
                distance_threshold = time_sec_threshold

            to_remove = set()
            while True:
                candidates = []
                remaining_idxs = list(sorted(set(range(len(file_system_preds))) - to_remove))

                if len(remaining_idxs) <= 1:
                    break

                for i in range(len(remaining_idxs)):
                    distance_before, distance_after = -1, -1

                    remaining_idx = remaining_idxs[i]
                    if i > 0:
                        before_idx = remaining_idxs[i - 1]
                        distance_before = file_system_preds[remaining_idx]['timestamp'] - file_system_preds[before_idx]['timestamp']
                    else:
                        before_idx = None

                    if i < len(remaining_idxs) - 1:
                        after_idx = remaining_idxs[i + 1]
                        distance_after = file_system_preds[after_idx]['timestamp'] - file_system_preds[remaining_idx]['timestamp']
                    else:
                        after_idx = None

                    # if the adjacent predictions are too close, we should consider removing it
                    if max(distance_before, distance_after) < distance_threshold:
                        if filtering == 'highest':
                            sort_key = -1 * float(file_system_preds[remaining_idx]['llr'])
                        elif filtering == 'lowest':
                            sort_key = float(file_system_preds[remaining_idx]['llr'])
                        elif filtering == 'most_similar':
                            sort_key = -1 * math.inf
                            if before_idx is not None:
                                sort_key = max(
                                    sort_key,
                                    abs(
                                        float(file_system_preds[remaining_idx]['llr']) -
                                        float(file_system_preds[before_idx]['llr'])
                                    )
                                )

                            if after_idx is not None:
                                sort_key = max(
                                    sort_key,
                                    abs(
                                        float(file_system_preds[remaining_idx]['llr']) -
                                        float(file_system_preds[after_idx]['llr'])
                                    )
                                )
                        else:
                            raise ValueError(f'Unknown filtering type: {filtering}')

                        candidates.append((sort_key, remaining_idx))

                if len(candidates) == 0:
                    break

                candidates = list(sorted(candidates))
                to_remove.add(candidates[0][1])

            filtered_system_preds.extend([
                file_system_preds[i] for i in range(len(file_system_preds)) if i not in to_remove
            ])

        return filtered_system_preds


    # refs: array of gold-label LDC references
    #   [
    #       {
    #           'file_id': the LDC file identifier for this changepoint
    #           'timestamp': the timestamp of the annotated changepoint
    #           'impact_scalar': the impact scalar of the annotated changepoint
    #           'type': one of audio / video / text
    #       }
    #   ]
    #   ex:
    #       [
    #           {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'impact_scalar': 4},
    #           {'file_id': 'M010015BY', 'timestamp': 1287.6, 'type': 'audio', 'impact_scalar': 2},
    #           {'file_id': 'M010029SP', 'timestamp': 288.0, 'type': 'text', 'impact_scalar': 1},
    #           {'file_id': 'M010005QD', 'timestamp': 90.2, 'type': 'video', 'impact_scalar': 5},
    #           {'file_id': 'M010019QD', 'timestamp': 90, 'type': 'text', 'impact_scalar': 5}
    #       ]
    # hyps: array of system predictions
    #   [
    #       {
    #           'file_id': the LDC file identifier for this changepoint
    #           'timestamp': the timestamp of the annotated changepoint
    #           'type': one of audio / video / text
    #           'llr': the log-likelihood ratio of the predicted changepoint
    #       }
    #   ]
    #   ex:
    #       [
    #           {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'llr': 1.5},
    #           {'file_id': 'M010015BY', 'timestamp': 1287.67, 'type': 'audio', 'llr': 1.5},
    #           {'file_id': 'M010029SP', 'timestamp': 288, 'type': 'text', 'llr': 1.5},
    #           {'file_id': 'M010005QD', 'timestamp': 90.2, 'llr': 1.5, 'type': 'video'},
    #           {'file_id': 'M010019QD', 'timestamp': 190, 'llr': 1.5, 'type': 'text'}
    #       ]
    # returns a dictionary with an AP score for each document type (audio, video, text)
    def calculate_average_precision(
            refs, hyps,
            text_char_threshold=100,
            time_sec_threshold=10,
            filtering='none'
    ):
        hyps = filter_system_preds(hyps, text_char_threshold, time_sec_threshold, filtering)

        # NIST uses non-zero values of "Class" to indicate annotations / predictions
        # in LDC's randomly selected annotation regions
        for ref in refs:
            ref['Class'] = ref['timestamp']
            ref['start'] = ref['timestamp']
            ref['end'] = ref['timestamp']

        for hyp in hyps:
            hyp['Class'] = hyp['timestamp']
            hyp['start'] = hyp['timestamp']
            hyp['end'] = hyp['timestamp']

        ref_df = pd.DataFrame.from_records(refs)
        hyp_df = pd.DataFrame.from_records(hyps)

        output_dir = 'tmp_scoring_%s' % os.getpid()
        os.makedirs(output_dir, exist_ok=True)

        score_cp(
            ref_df, hyp_df,
            delta_cp_text_thresholds=[text_char_threshold],
            delta_cp_time_thresholds=[time_sec_threshold],
            output_dir=output_dir
        )

        APs, score_df = {}, pd.read_csv(
            os.path.join(output_dir, 'scores_by_class.tab'), delimiter='\t'
        )
        for _, row in score_df[score_df['metric'] == 'AP'].iterrows():
            APs[row['genre']] = float(row['value'])

        shutil.rmtree(output_dir)

        return APs
    
    # load and preprocess dataset
    data_path = args.dataset_path
    with open(data_path, 'rb') as handle:
        dataset = pickle.load(handle)

    # delete file without jpgs
    keys_to_remove = []
    for key in dataset.keys():
        if dataset[key]['data_type'] !='video':
            keys_to_remove.append(key)
        elif dataset[key]['processed'] == False:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del dataset[key]

    # train val test split
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}
    final_eval_dataset = {}
    for key in dataset.keys():
        if 'INTERNAL_TRAIN' in dataset[key]['splits']:
            train_dataset.update({key:dataset[key]})
        if 'EVALUATION_LDC2023E07' in dataset[key]['splits']:
            final_eval_dataset.update({key:dataset[key]})
        if 'INTERNAL_VAL' in dataset[key]['splits']:
            val_dataset.update({key:dataset[key]})
        if 'INTERNAL_TEST' in dataset[key]['splits']:
            test_dataset.update({key:dataset[key]})
            
    print(f'Inference batch size {args.batch_size}')

    print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset), len(final_eval_dataset))
    
    

    dataset_dict = {'EVALUATION_LDC2023E07_whisper':final_eval_dataset,
                   'EVALUATION_LDC2023E07_azure':final_eval_dataset,
                   'EVALUATION_LDC2023E07_wav2vec':final_eval_dataset,
                   'INTERNAL_VAL':val_dataset,
                   'INTERNAL_TEST':test_dataset}
    dataset_name = args.dataset_name
    if 'EVALUATION_LDC2023E07' not in dataset_name:
        transcribe_name = 'whisper'
    else:
        transcribe_name = dataset_name.split('_')[2]
    model_path = args.model_save_path
    if args.part is not None:
        save_path = os.path.join(args.result_save_path, f'inference_result_{dataset_name}_{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_part_{args.part}')
    else:
        save_path = os.path.join(args.result_save_path, f'inference_result_{dataset_name}_{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}')
    print(f'Loading model from {model_path}')
    print(f'Result save to {save_path}')
    print(f'Start inference on {dataset_name}')
    val_dataset_with_model(model_path ,dataset_dict[dataset_name] , save_path, False, transcribe_name,args)