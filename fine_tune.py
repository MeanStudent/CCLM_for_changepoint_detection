from useful_functions import *
import argparse
# remember to set GPU
config = yaml.load(open('configs/Pretrain_4m.yaml', 'r'), Loader=yaml.Loader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# data augumentations
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

train_transform = transforms.Compose([
    transforms.CenterCrop(config['image_res'],# scale=(0.5, 1.0),
                                 #interpolation=InterpolationMode.BICUBIC
                         ),
    #transforms.RandomHorizontalFlip(),
    RandomAugment(2, 7, isPIL=True, augs=['AutoContrast','Identity', 'Brightness', 'Sharpness']),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

device = 'cuda'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dsp", "--dataset_path", default = '/mnt/swordfish-pool2/ccu/amith-cache.pkl')
    parser.add_argument("-tbs", "--train_batch_size", type = int, default = 50)
    parser.add_argument("-ebs", "--eval_batch_size", type = int, default = 512)
    parser.add_argument("-ne", "--num_epochs", type = int, default = 20)
    parser.add_argument("-uc", "--use_context", type = int, default = 1)
    parser.add_argument("-lr", "--lr", type = float, default = 0.00002)
    parser.add_argument("-tn", "--transcribe_name", type = str, default = 'whisper')
    parser.add_argument("-gpu", "--gpu_index", required=True)
    parser.add_argument("-npr", "--neg_pos_rate", type=float, required= True)
    parser.add_argument("-tchm", "--trained_cls_head_model", type = str, default = '/mnt/swordfish-pool2/kh3074/neg_pos_rate2/trained_cls_head_model/model_tuned_epoch_44')
    parser.add_argument("-msd", "--model_save_dir",  default = '/mnt/swordfish-pool2/kh3074/neg_pos_rate2/saved_models')
    parser.add_argument("-rsd", "--result_save_dir", default = '/mnt/swordfish-pool2/kh3074/neg_pos_rate2/evaluate_results')
    
    args = parser.parse_args()
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    print(f'using {args.gpu_index}')
    
    
    data_path = args.dataset_path
    print('Start loading and preprocessing dataset!')
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
    
    train_csv = construct_dataset_csv(train_dataset,args.transcribe_name,use_context = args.use_context)
    val_csv = construct_dataset_csv(val_dataset,args.transcribe_name,use_context = args.use_context)
    test_csv = construct_dataset_csv(test_dataset,args.transcribe_name,use_context = args.use_context)
    
    
    print('Start loading model')
    # parameters
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    num_epoch = args.num_epochs
    custom_lr = args.lr
    # original 0.0001

    # load model
    my_nvlr_model = NLVRModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    if args.use_context:
        special_tokens = ['<Pre_Context>', '<Post_Context>']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        my_nvlr_model.text_encoder.resize_token_embeddings(len(tokenizer))
    
    my_nvlr_model = nn.DataParallel(my_nvlr_model,device_ids=list(range(len(args.gpu_index.split(',')))))
    my_nvlr_model.to(device)
    checkpoint_ldc = torch.load(args.trained_cls_head_model, map_location=device)
    my_nvlr_model.load_state_dict(checkpoint_ldc) 
    my_nvlr_model.to(device)
 

    
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})

    # training parameters 
    world_size = utils.get_world_size()

    arg_opt = utils.AttrDict(config['optimizer'])
    arg_opt['lr'] = custom_lr
    optimizer = create_optimizer(arg_opt, my_nvlr_model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(len(train_dataset)/(train_batch_size*world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)
    log = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    print_freq = 6
    train_accuracys = []
    train_recalls = []
    train_precisions = []
    train_aucs = []
    train_auc_precisions = []
    train_auc_recalls = []
    val_accuracys = []
    val_recalls = []
    val_precisions = []
    val_aucs = []
    val_auc_precisions = []
    val_auc_recalls = []

    top_models = [] # models at current top performance
    model_save_dir = args.model_save_dir
    log_save_dir = args.result_save_dir


    print('Start training!!')
    for epoch in range(num_epoch):
        # start new epoch of training
        my_nvlr_model.train()
        for param in my_nvlr_model.parameters():
            param.requires_grad = True
        for param in my_nvlr_model.module.cls_head.parameters():
        #for param in my_nvlr_model.cls_head.parameters(): 
            param.requires_grad = True

        train_data_loader = create_down_sample_dataloader(train_csv, epoch, train_batch_size, 
                                  train_transform,args.neg_pos_rate,evaluation = False) # use epoch as random seed so that every epoch will have fixed positive sample and different negative sample
        header = 'Train Epoch: [{}]'.format(epoch) 
        for i, (image0, image1, text, targets) in enumerate(metric_logger.log_every(train_data_loader, print_freq, header)):
            images = torch.cat([image0, image1], dim=0)
            images, targets = images.to(device), targets.to(device)   

            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

            loss = my_nvlr_model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)
            loss = loss.mean() # aggregate loss from different GPUs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())

        # start evaluation
        my_nvlr_model.eval() 
        # create eval_Dataloader with fix seed for val dataset
        print('start eval on train dataset ---------------------------------------------')
        train_eval_data_loader = create_down_sample_dataloader(train_csv, 2333, eval_batch_size, 
                                  val_transform,args.neg_pos_rate,evaluation = True)
        eval_train_df = eval_on_dataset(my_nvlr_model,train_eval_data_loader,device,tokenizer,log_save_dir)
        train_recall, train_precision, train_accuracy, train_pr_auc, train_precision_auc, train_recall_auc = calculate_matrix(eval_train_df)

        train_accuracys.append(train_accuracy)
        train_recalls.append(train_recall)
        train_precisions.append(train_precision)
        train_aucs.append(train_pr_auc)
        train_auc_precisions.append(train_precision_auc)
        train_auc_recalls.append(train_recall_auc)


        print('start eval on validation dataset ---------------------------------------------')
        val_data_loader = create_down_sample_dataloader(val_csv, 2333, eval_batch_size, 
                                  val_transform,args.neg_pos_rate,evaluation = True) # fix seed 2333
        eval_val_df = eval_on_dataset(my_nvlr_model,val_data_loader,device,tokenizer,log_save_dir)
        val_recall, val_precision, val_accuracy, val_pr_auc, val_precision_auc, val_recall_auc = calculate_matrix(eval_val_df)

        val_accuracys.append(val_accuracy)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        val_aucs.append(val_pr_auc)
        val_auc_precisions.append(val_precision_auc)
        val_auc_recalls.append(val_recall_auc)


        # if auc of val dataset is the top 3, then save the model
        if len(top_models) < 3:
            top_models.append((f'model_tuned_epoch_{epoch}', val_pr_auc))
        else:
            top_models.append((f'model_tuned_epoch_{epoch}', val_pr_auc))
            top_models.sort(key=lambda x: x[1], reverse=True)

            if os.path.exists(os.path.join(model_save_dir,top_models[-1][0])):
                os.remove(os.path.join(model_save_dir,top_models[-1][0]))
            top_models.pop(3)
            for i in range(3):
                if os.path.exists(os.path.join(model_save_dir,top_models[i][0])):
                    pass
                else:
                    torch.save(my_nvlr_model.state_dict(), os.path.join(model_save_dir,top_models[i][0]))
        torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())     
        log.append({k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()})

        #vis
        train_result_df = pd.DataFrame({'train_accuracys':train_accuracys,
                             'train_recalls':train_recalls ,
                             'train_precisions':train_precisions ,
                             'train_aucs':train_aucs ,
                             'train_auc_precisions':train_auc_precisions ,
                             'train_auc_recalls':train_auc_recalls ,
                             'val_accuracys':val_accuracys ,
                             'val_recalls':val_recalls ,
                             'val_precisions':val_precisions ,
                             'val_aucs':val_aucs ,
                             'val_auc_precisions':val_auc_precisions ,
                             'val_auc_recalls':val_auc_recalls })


        train_result_df.to_csv(os.path.join(log_save_dir,'train_epoch_for_fine_tune'))
    print('top saved models:')
    print(top_models)
