############################################################################################################
from re import template
from transformers import T5ForConditionalGeneration, T5Tokenizer,AutoTokenizer
from mnli_dataloader import MNLIDataLoader
from mnli_dataset import MNLIDataset
from mnli_model import MNLIT5
from roberta_dataset import RobertaDataset
from bert import Bert
from bert_prompt import BertPrompt
############################################################################################################

import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import AdamW, Adafactor

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import is_first_worker, DistributedEvalSampler,set_dist_args, optimizer_to
from contextlib import nullcontext # from contextlib import suppress as nullcontext # for python < 3.7
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import random
import numpy as np
from tqdm import tqdm
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def test(args, model, test_loader, device, tokenizer):
    total=0
    right=0
    for test_batch in tqdm(test_loader, disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            if args.model=="t5":
                if args.original_t5:
                    output_sequences = model.module.t5.generate(
                        input_ids=test_batch['input_ids'].to(device),
                        attention_mask=test_batch['attention_mask'].to(device),
                        do_sample=False, # disable sampling to test if batching affects output
                    )
                    batch_result = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                    #print(batch_result)
                    # print(batch_result)
                    true_result = test_batch["raw_label"]
                    total += len(true_result)
                    for br, tr in zip(batch_result, true_result):
                        if br == tr:
                            right += 1
                else:
                    batch_score = model(
                            input_ids=test_batch['input_ids'].to(device), 
                            attention_mask=test_batch['attention_mask'].to(device),
                            hypothesis_ids=test_batch['hypothesis_ids'].to(device), 
                            premise_ids=test_batch['premise_ids'].to(device),
                            hypothesis_attention_mask=test_batch['hypothesis_attention_mask'].to(device),
                            premise_attention_mask=test_batch['premise_attention_mask'].to(device),
                            labels=test_batch['labels'].to(device)
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=test_batch['label_id'].to(device)
                    total+=len(label)
            elif args.model=="roberta":
                if args.task=="prompt_classification":
                    batch_score,_ = model(
                            masked_token_pos=test_batch['mask_pos'].to(device),
                            input_ids=test_batch['input_ids'].to(device), 
                            attention_mask=test_batch['attention_mask'].to(device),
                            hypothesis_ids=test_batch['hypothesis_ids'].to(device), 
                            premise_ids=test_batch['premise_ids'].to(device),
                            hypothesis_attention_mask=test_batch['hypothesis_attention_mask'].to(device),
                            premise_attention_mask=test_batch['premise_attention_mask'].to(device)
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=test_batch['label_id'].to(device)
                    total+=len(label)
                else:
                    batch_score,_ = model(
                            input_ids=test_batch['input_ids'].to(device), 
                            attention_mask=test_batch['attention_mask'].to(device)
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=test_batch['label_id'].to(device)
                    total+=len(label)
                right+=torch.eq(predict,label).sum()
    return total, int(right)


def dev(args, model, dev_loader, device, tokenizer):
    total=0
    right=0
    for dev_batch in tqdm(dev_loader, disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            if args.model=="t5":
                if args.original_t5:
                    output_sequences = model.module.t5.generate(
                        input_ids=dev_batch['input_ids'].to(device),
                        attention_mask=dev_batch['attention_mask'].to(device),
                        do_sample=False, # disable sampling to test if batching affects output
                    )
                    batch_result = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                    #print(batch_result)
                    # print(batch_result)
                    true_result = dev_batch["raw_label"]
                    total += len(true_result)
                    for br, tr in zip(batch_result, true_result):
                        if br == tr:
                            right += 1
                else:
                    batch_score = model(
                            input_ids=dev_batch['input_ids'].to(device), 
                            attention_mask=dev_batch['attention_mask'].to(device),
                            hypothesis_ids=dev_batch['hypothesis_ids'].to(device), 
                            premise_ids=dev_batch['premise_ids'].to(device),
                            hypothesis_attention_mask=dev_batch['hypothesis_attention_mask'].to(device),
                            premise_attention_mask=dev_batch['premise_attention_mask'].to(device),
                            labels=dev_batch['labels'].to(device)
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=dev_batch['label_id'].to(device)
                    total+=len(label)
            elif args.model=="roberta":
                if args.task=="prompt_classification":
                    batch_score,_ = model(
                        masked_token_pos=dev_batch['mask_pos'].to(device),
                            input_ids=dev_batch['input_ids'].to(device), 
                            attention_mask=dev_batch['attention_mask'].to(device),
                            hypothesis_ids=dev_batch['hypothesis_ids'].to(device), 
                            premise_ids=dev_batch['premise_ids'].to(device),
                            hypothesis_attention_mask=dev_batch['hypothesis_attention_mask'].to(device),
                            premise_attention_mask=dev_batch['premise_attention_mask'].to(device),
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=dev_batch['label_id'].to(device)
                    total+=len(label)
                else:
                    batch_score,_ = model(
                            input_ids=dev_batch['input_ids'].to(device), 
                            attention_mask=dev_batch['attention_mask'].to(device),
                            )
                    predict=torch.argmax(batch_score,dim=1)
                    label=dev_batch['label_id'].to(device)
                    total+=len(label)
                right+=torch.eq(predict,label).sum()
    return total, int(right)


def batch_to_device(batch, device):
    device_batch = {}
    for key, value in batch.items():
        device_batch[key] = value.to(device)
    return device_batch
            

def train(args, model, loss_fn, m_optim, m_scheduler, train_loader, dev_loader, test_loader,device, train_sampler=None, tokenizer=None):
    best_mes = 0.0
    global_step = 0 # steps that outside epoches
    force_break = False
    for epoch in range(args.epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            # print("Before: global step {}, rank {}".format(global_step, args.local_rank))
            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext
            with sync_context():
                if args.model=="t5":
                    batch_score,batch_loss = model(
                        input_ids=train_batch['input_ids'].to(device), 
                        attention_mask=train_batch['attention_mask'].to(device),
                        hypothesis_ids=train_batch['hypothesis_ids'].to(device), 
                        premise_ids=train_batch['premise_ids'].to(device),
                        hypothesis_attention_mask=train_batch['hypothesis_attention_mask'].to(device),
                        premise_attention_mask=train_batch['premise_attention_mask'].to(device),
                        labels=train_batch['labels'].to(device)
                        )
                elif args.model=="roberta":
                    if args.task=="prompt_classification":
                        batch_score,_=model(
                            masked_token_pos=train_batch['mask_pos'].to(device),
                            input_ids=train_batch['input_ids'].to(device), 
                            attention_mask=train_batch['attention_mask'].to(device),
                            hypothesis_ids=train_batch['hypothesis_ids'].to(device), 
                            premise_ids=train_batch['premise_ids'].to(device),
                            hypothesis_attention_mask=train_batch['hypothesis_attention_mask'].to(device),
                            premise_attention_mask=train_batch['premise_attention_mask'].to(device)
                        )
                    elif args.task=="classification":
                        batch_score,_=model(
                            input_ids=train_batch['input_ids'].to(device), 
                            attention_mask=train_batch['attention_mask'].to(device)
                        )
            with sync_context():
                if not args.model is "t5":
                    batch_loss = loss_fn(batch_score, train_batch['label_id'].to(device))

            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            with sync_context():
                batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                m_optim.step()
                if m_scheduler is not None:
                    m_scheduler.step()
                m_optim.zero_grad()
              

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    
                    if args.local_rank in [-1, 0]:
                        logger.info("training gpu {}:,  global step: {}, local step: {}, loss: {}".format(args.local_rank,global_step+1, step+1, avg_loss/args.logging_step))
                        if args.tb is not None:
                            args.tb.add_scalar("loss", avg_loss/args.logging_step, global_step + 1)
                            args.tb.add_scalar("epochs", epoch + 1, global_step + 1)
                    avg_loss = 0.0 

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):
                    if not args.no_dev:
                        model.eval()
                        with torch.no_grad():
                            total,right=dev(args, model, dev_loader, device, tokenizer)
                        model.train()
                        if not os.path.exists(args.res):
                                os.makedirs(args.res)
                        with open(args.res+'results.jsonl','a') as f:
                            f.write(json.dumps({'total':total,'right':right}))
                            f.write('\n')
                        if args.local_rank != -1:
                            dist.barrier()
                        if args.local_rank in [-1,0]:
                            with open(args.res+'results.jsonl','r') as f:
                                r,t=0,0
                                for line in f:
                                    r+=eval(line)['right']
                                    t+=eval(line)['total']
                                    print(r,t)
                                mes=r/t
                            if args.tb is not None:
                                args.tb.add_scalar("dev_acc", mes, global_step + 1)
                            os.remove(args.res+'results.jsonl')#logger.info('save_model at step {}'.format(global_step+1))
                            if not os.path.exists(args.save):
                                    os.makedirs(args.save)
                            if mes>best_mes:
                                best_mes=mes
                                ls=os.listdir(args.save)
                                for i in ls:
                                    item_path=os.path.join(args.save,i)
                                    #print("remove {}".format(item_path))
                                # logger.info('remove_model at step {}'.format(global_step+1))
                                # logger.info('save model')
                                # os.remove(item_path)
                            logger.info("global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))
                    if hasattr(model, "module"):
                        torch.save(model.module.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                    else:
                        torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                
                global_step += 1

                if args.max_steps is not None and global_step == args.max_steps:
                    force_break = True
                    break

            if args.local_rank != -1:
                dist.barrier()

        if args.local_rank != -1:
            dist.barrier()
        if force_break:
            break
    if not args.no_test:
        if args.local_rank != -1:
            dist.barrier()
            if not os.path.exists(args.test_res):
                os.makedirs(args.test_res)
            logger.info("load best checkpoint....")
            dist.barrier()
            for file in os.listdir(args.save):
                checkpoint=os.path.join(args.save,file)
                state=torch.load(checkpoint,map_location=device)
                model.module.load_state_dict(state)
            dist.barrier()
            logger.info("doing inference.... at gpu:{}".format(args.local_rank))
            model.eval()
            with torch.no_grad():
                total,right = test(args, model,test_loader,device,tokenizer=tokenizer)
            logger.info("inference finished.... at gpu:{}".format(args.local_rank))
            dist.barrier()
            if args.local_rank in [-1,0]:
                if os.path.exists(args.test_res+"test_results.jsonl"):
                    os.remove(args.test_res+"test_results.jsonl")
            dist.barrier()
            with open(args.test_res+"test_results.jsonl",'a') as f:
                f.write(json.dumps({'total':total,'right':right}))
                f.write('\n')
            if args.local_rank != -1:
                dist.barrier()
            if args.local_rank in [-1,0]:
                with open(args.test_res+"test_results.jsonl",'r+') as f:
                    r,t=0,0
                    for line in f:
                        r+=eval(line)['right']
                        t+=eval(line)['total']
                        print(r,t)
                    mes=r/t
                    logger.info("test accuracy:{}".format(mes))
                    f.write("\ntest accuracy:{}".format(mes))
    return 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-train', type=str, default='./data/train_toy.jsonl')
    parser.add_argument('-dev', type=str, default='./data/dev_toy.jsonl')
    parser.add_argument('-test', type=str, default='./data/dev_toy.jsonl')
    parser.add_argument('--model', type=str, default='t5')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-test_res', type=str, default='./results/bert.trec')
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float,help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('-n_warmup_steps',type=int,default=0)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-right',type=int,default=0)
    parser.add_argument("-task",type=str,default="classification")
    parser.add_argument("-template",type=str,default="")
    parser.add_argument("--soft_prompt", action="store_true")
    parser.add_argument("--infix",type=str,default=None)
    parser.add_argument("--suffix",type=str,default=None)
    parser.add_argument("--prefix",type=str,default=None)
    parser.add_argument("--original_t5", action="store_true")
    parser.add_argument("--no_dev", action="store_true",default=False)
    parser.add_argument("--no_test", action="store_true",default=False)
    args = parser.parse_args()
    set_seed(13)
    set_dist_args(args) # get local cpu/gpu device
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
        args.tb = writer
    else:
        args.tb = None

    tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    if args.model=="t5":
        logger.info('reading training t5 data...')
        train_set=MNLIDataset(original_t5=args.original_t5,dataset=args.train,tokenizer=tokenizer,template=args.template)
        dev_set=None
        test_set=None
        if not args.no_dev:
            logger.info('reading dev t5 data...')
            dev_set=MNLIDataset(original_t5=args.original_t5,dataset=args.dev,tokenizer=tokenizer,template=args.template)
        if not args.no_test:
            logger.info('reading test t5 data...')
            test_set=MNLIDataset(original_t5=args.original_t5,dataset=args.dev,tokenizer=tokenizer,template=args.template)
        model = MNLIT5(args.original_t5,args.pretrain,args.soft_prompt,args.prefix,args.infix,args.suffix)
    elif args.model=="roberta":
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        logger.info('reading training roberta data...')
        train_set=RobertaDataset(task=args.task,max_input=args.max_input,dataset=args.train,tokenizer=tokenizer,template=args.template)
        logger.info('reading dev roberta data...')
        dev_set=RobertaDataset(task=args.task,max_input=args.max_input,dataset=args.dev,tokenizer=tokenizer,template=args.template)
        logger.info('reading test roberta data...')
        test_set=RobertaDataset(task=args.task,max_input=args.max_input,dataset=args.test,tokenizer=tokenizer,template=args.template)
        if args.task=="prompt_classification":
            model = BertPrompt(pretrained=args.pretrain,soft_prompt=args.soft_prompt,prefix=args.prefix,suffix=args.suffix)
            model._model.resize_token_embeddings(len(tokenizer))
        else:
            model=Bert(pretrained=args.pretrain,num_labels=3)
    if args.local_rank != -1:
        
        train_sampler = DistributedSampler(train_set)
        train_loader = MNLIDataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            sampler=train_sampler
        )
        dev_sampler=None
        dev_loader=None
        test_sampler=None
        test_loader=None
        if dev_set is not None:
            dev_sampler = DistributedEvalSampler(dev_set)
            dev_loader = MNLIDataLoader(
                dataset=dev_set,
                batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
                shuffle=False,
                num_workers=8,
                sampler=dev_sampler
            )
        if test_set is not None:
            test_sampler = DistributedEvalSampler(test_set)
            test_loader = MNLIDataLoader(
                dataset=test_set,
                batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
                shuffle=False,
                num_workers=8,
                sampler=test_sampler
            )
        dist.barrier()


    device = args.device
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    if args.checkpoint is not None:
        st=torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(st)
    loss_fn.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    model.zero_grad()
    model.train()
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == "adafactor":
        m_optim = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    if args.optimizer.lower() == "adafactor":
        m_scheduler = None
    else:
        if args.local_rank == -1:
            m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size if args.max_steps is None else args.max_steps)
        else:
            m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.world_size*args.gradient_accumulation_steps) if args.max_steps is None else args.max_steps)

    if m_optim is not None:
        optimizer_to(m_optim,device)

    logger.info(args)
    train(args, model, loss_fn, m_optim, m_scheduler,  train_loader, dev_loader,test_loader,device, train_sampler=train_sampler, tokenizer=tokenizer)
    if args.local_rank != -1:
        dist.barrier()
if __name__ == "__main__":
    main()
    os._exit(0)
