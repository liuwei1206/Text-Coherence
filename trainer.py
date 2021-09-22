# author = liuwei
# date = 2021-09-21
# If any question, please contact the email "willie1206@163.com"

import logging
import json
import math
import os
import random
import time
from packaging import version
import pickle

import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from contextlib import contextmanager

from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.configuration_albert import AlbertConfig
# from transformers.models.xlnet.configuration_xlnet import XLNetConfig
# from transformers.models.xlnet.tokenization_xlnet import XLNetTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from args.doc_parser import get_argparse
from model.pretrained_for_text_classification import PretrainedForTextClassification
from module.sampler import SequentialDistributedSampler
from module.progress_bar import ProgressBar
from feature.gcdc_dataset import GCDCDataset
from feature.vocab import ItemVocabFile, ItemVocabArray
from function.metrics import accuracy_score
from function.writer import save_preds_for_text_classification

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("No Tensorboard Found!!!")


# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logfile = './data/log/gcdc_{}.txt'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode='train'):
    """
    generator datasetloader for training.
    Note that: for training, we need random sampler, same to shuffle
               for eval or predict, we need sequence sampler, same to no shuffle
    Args:
        dataset:
        args:
        mode: train or non-train
    """
    print("Dataset length: ", len(dataset))
    if mode == 'train':
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.per_gpu_train_batch_size,
            shuffle=False, # Randomsampler already contains shuffle
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )
    else:
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.per_gpu_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )

    return data_loader

def get_optimizer(model, args, num_training_steps):
    """
    Setup the optimizer and the learning rate scheduler
    we provide a reasonable default that works well
    If you want to use something else, you can pass a tuple in the Trainer's init,
    or override this method in a subclass.
    """
    # big_lr_params = ["word_embedding", "attn_w", "word_transform", "word_word_weight", "hidden2tag",
    #              "lstm", "crf"]

    big_lr_params = []
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in big_lr_params)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in big_lr_params)],
            "lr": 0.0001
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def print_log(logs, epoch, global_step, eval_type, tb_writer, iterator=None):
    """print training informations"""
    if epoch is not None:
        logs['epoch'] = epoch
    if global_step is None:
        global_step = 0
    if eval_type in ["Dev", "Test"]:
        print("#############  %s's result  #############"%(eval_type))
    if tb_writer:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    '"%s" of type %s for key "%s" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.",
                    v,
                    type(v),
                    k,
                )
        tb_writer.flush()

    output = {**logs, **{"step": global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)


def train(model, args, train_dataset, dev_dataset, test_dataset, tb_writer, model_path=None):
    """
    train the model
    Args:
        model: designed model
        train_dataset: instance of a dataset
        dev_dataset: development dataset
        test_dataset: test dataset
        tb_writer: write into summary file
        model_path: path to load model
    """
    ## 1.prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    if args.max_steps > 0:
        t_total = args.max_steps
        num_train_epochs = (
                args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs

    ## 2.optimizer and model
    optimizer, scheduler = get_optimizer(model, args, t_total)

    # Check if saved optimizer or scheduler states exist
    if (model_path is not None
        and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(
            torch.load(os.path.join(model_path, "optimizer.pt"), map_location=args.device)
        )
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    ## 3.begin train
    total_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    if args.local_rank == 0 or args.local_rank == -1:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if model_path is not None: # load checkpoint and continue training
        try:
            global_step = int(model_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
            )
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            global_step = 0
            logger.info("  Starting fine-tuning.")

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        # for step, batch in enumerate(epoch_iterator):
        pbar = ProgressBar(n_total=len(train_dataloader), desc="Training")
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                # Skip past any already trained steps if resuming training
                steps_trained_in_current_epoch -= 1
                continue
            model.train()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"doc_token_ids": batch[0], "doc_segment_ids": batch[1],
                      "doc_attention_mask": batch[2],
                      "doc_para_token_ids": batch[3], "doc_para_segment_ids": batch[4],
                      "doc_para_attention_mask": batch[5], "para_attention_mask": batch[6],
                      "doc_sent_token_ids": batch[7], "doc_sent_segment_ids": batch[8],
                      "doc_sent_attention_mask": batch[9], "sent_attention_mask": batch[10],
                      "labels": batch[11], "mode": args.doc_type, "flag": "Train"}

            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            pbar(step, {'loss': loss.item()}) # print the training progress bar

            ## update gradient
            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    ((step + 1) == len(epoch_iterator)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                ## logger and evaluate
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    logs["loss"] = (tr_loss - logging_loss) / args.logging_steps
                    # backward compatibility for pytorch schedulers
                    logs["learning_rate"] = (
                        scheduler.get_last_lr()[0]
                        if version.parse(torch.__version__) >= version.parse("1.4")
                        else scheduler.get_lr()[0]
                    )
                    logging_loss = tr_loss
                    if args.local_rank == 0 or args.local_rank == -1:
                        print_log(logs, epoch, global_step, "", tb_writer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # save after each epoch
        output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # evaluate after each epoch
        if args.evaluate_during_training:
            # for dev
            metrics, _ = evaluate(model, args, dev_dataset, global_step, description="Dev", write_file=True)
            print_log(metrics, epoch, global_step, "Dev", tb_writer)

            # for test
            metrics, _ = evaluate(model, args, test_dataset, global_step, description="Test", write_file=True)
            print_log(metrics, epoch, global_step, "Test", tb_writer)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # save the last one
    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    print("global_step: ", global_step)
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    return global_step, tr_loss / global_step

def evaluate(model, args, dataset, global_step, description="dev", write_file=False):
    """
    evaluate the model's performance
    """
    dataloader = get_dataloader(dataset, args, mode='dev')
    if (not args.do_train) and (not args.no_cuda):
        model = model.cuda()

    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", batch_size)
    model.eval()

    eval_losses = []
    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_attention_mask = None

    # for batch in tqdm(dataloader, desc=description):
    pbar = ProgressBar(n_total=len(dataloader), desc="Evaluation")
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"doc_token_ids": batch[0], "doc_segment_ids": batch[1],
                  "doc_attention_mask": batch[2],
                  "doc_para_token_ids": batch[3], "doc_para_segment_ids": batch[4],
                  "doc_para_attention_mask": batch[5], "para_attention_mask": batch[6],
                  "doc_sent_token_ids": batch[7], "doc_sent_segment_ids": batch[8],
                  "doc_sent_attention_mask": batch[9], "sent_attention_mask": batch[10],
                  "labels": batch[11], "mode": args.doc_type, "flag": "Eval"}
        with torch.no_grad():
                outputs = model(**inputs)
                preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[11].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        attention_mask = batch[2].detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_attention_mask = attention_mask
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)

    ## calculate metrics
    acc = accuracy_score(all_label_ids, all_predict_ids)
    metrics = {}
    metrics['acc'] = acc

    ## write labels into file
    if write_file:
        file_path = os.path.join(args.output_dir, "{}-{}-{}.txt".format(args.model_type, description, str(global_step)))
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
        save_preds_for_text_classification(all_input_ids, attention_mask, tokenizer, all_label_ids, all_predict_ids, file_path)

    return metrics, (all_label_ids, all_predict_ids)

def main():
    args = get_argparse().parse_args()
    args.no_cuda = not torch.cuda.is_available()

    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = SummaryWriter(log_dir=args.logging_dir)
    set_seed(args.seed)

    ## 1.prepare data
    domain = "xxx_"
    train_data_file = os.path.join(args.data_dir, domain + "train.json")
    dev_data_file = os.path.join(args.data_dir, domain + "test.json")
    test_data_file = os.path.join(args.data_dir, domain + "test.json")

    if args.model_type == "Bert":
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
        config = BertConfig.from_pretrained(args.config_name)
    elif args.model_type == "Roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.vocab_file)
        config = RobertaConfig.from_pretrained(args.config_name)
    elif args.model_type == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained(args.vocab_file)
        config = AlbertConfig.from_pretrained(args.config_name)
    elif args.model_type == "XLNet":
        tokenizer = XLNetTokenizer.from_pretrained(args.vocab_file)
        config = XLNetConfig.from_pretrained(args.config_name)

    params = {}
    params['model_type'] = args.model_type
    params['doc_type'] = args.doc_type
    params['para_pooling_type'] = args.para_pooling_type
    params['para_d_model'] = config.hidden_size
    params['para_nhead'] = config.num_attention_heads
    params['para_num_layers'] = 2
    params['sent_pooling_type'] = args.sent_pooling_type
    params['para_d_model'] = config.hidden_size
    params['para_nhead'] = config.num_attention_heads
    params['sent_num_layers'] = 2
    params['doc_pooling_type'] = args.doc_pooling_type
    params['num_labels'] = 3

    model = PretrainedForTextClassification(
        config=config,
        params=params
    )

    if not args.no_cuda:
        model = model.cuda()
    args.label_size = 3
    dataset_params = {
        'tokenizer': tokenizer,
        'max_doc_length': args.max_doc_length,
        'max_para_num': args.max_para_num,
        'max_para_length': args.max_para_length,
        'max_sent_num': args.max_sent_num,
        'max_sent_length': args.max_sent_length,
    }

    if args.do_train:
        train_dataset = GCDCDataset(train_data_file, params=dataset_params)
        dev_dataset = GCDCDataset(dev_data_file, params=dataset_params)
        test_dataset = GCDCDataset(test_data_file, params=dataset_params)
        args.model_name_or_path = None
        train(model, args, train_dataset, dev_dataset, test_dataset, tb_writer)

    if args.do_eval:
        logger.info("*** Dev Evaluate ***")
        dev_dataset = GCDCDataset(dev_data_file, params=dataset_params)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]
        eval_output, _ = evaluate(model, args, dev_dataset, global_steps, "dev", write_file=True)
        eval_output["global_steps"] = global_steps
        print("Dev Result: acc: %.4f\n"%(eval_output['acc']))


    # return eval_output
    if args.do_predict:
        logger.info("*** Test Evaluate ***")
        test_dataset = GCDCDataset(test_data_file, params=dataset_params, do_shuffle=False)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]
        eval_output, _ = evaluate(model, args, test_dataset, global_steps, "test", write_file=True)
        eval_output["global_steps"] = global_steps
        print("Test Result: acc: %.4f\n" %(eval_output['acc']))

if __name__ == "__main__":
    main()
