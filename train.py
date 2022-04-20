import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import T5ForConditionalGeneration,Adafactor
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import src as om
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import is_first_worker, DistributedEvalSampler, merge_resfile, set_dist_args, optimizer_to
from contextlib import nullcontext # from contextlib import suppress as nullcontext # for python < 3.7
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import random
import numpy as np
from tqdm import tqdm
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# writer = SummaryWriter(log_dir='logs')

def test(args, model, metric, test_loader, device,tokenizer):
    rst_dict = {}
    for test_batch in tqdm(test_loader, disable=args.local_rank not in [-1, 0]):
        if args.model=="t5":
            query_id, doc_id, label = test_batch['query_id'], test_batch['doc_id'], test_batch['label_id']
        else:
            query_id, doc_id, label = test_batch['query_id'], test_batch['doc_id'], test_batch['label']
        with torch.no_grad():
            #if args.original_t5:
            #    batch_score,_ = model(                        
            #            input_ids=test_batch['input_ids'].to(device), 
            #            attention_mask=test_batch['attention_mask'].to(device), 
            #            labels=test_batch["labels"].to(device),
            #            return_dict=True
            #        )
            if args.model== 't5':
                batch_score,_=model(
                        input_ids=test_batch['input_ids'].to(device), 
                        attention_mask=test_batch['attention_mask'].to(device),
                        query_ids=test_batch['query_ids'].to(device), 
                        doc_ids=test_batch['doc_ids'].to(device),
                        query_attention_mask=test_batch['query_attention_mask'].to(device),
                        doc_attention_mask=test_batch['doc_attention_mask'].to(device),
                        labels=test_batch['labels'].to(device)
                )
            elif args.model == 'bert':
                if args.task.startswith("prompt"):
                    batch_score, _ = model(
                        test_batch['input_ids'].to(device),  test_batch['query_ids'].to(device), test_batch['doc_ids'].to(device), 
                        test_batch['mask_pos'].to(device),
                        test_batch['input_mask'].to(device), test_batch['query_input_mask'].to(device),test_batch['doc_input_mask'].to(device),
                        test_batch['segment_ids'].to(device)
                        )
                else:
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device), test_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                if args.task.startswith("prompt"):
                    batch_score, _ = model(
                        test_batch['input_ids'].to(device), test_batch['query_ids'].to(device), test_batch['doc_ids'].to(device), 
                        test_batch['mask_pos'].to(device), 
                        test_batch['input_mask'].to(device),test_batch['query_input_mask'].to(device),test_batch['doc_input_mask'].to(device)
                        )
                else:
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device))

            elif args.model == 'edrm':
                batch_score, _ = model(test_batch['query_wrd_idx'].to(device), test_batch['query_wrd_mask'].to(device),
                                       test_batch['doc_wrd_idx'].to(device), test_batch['doc_wrd_mask'].to(device),
                                       test_batch['query_ent_idx'].to(device), test_batch['query_ent_mask'].to(device),
                                       test_batch['doc_ent_idx'].to(device), test_batch['doc_ent_mask'].to(device),
                                       test_batch['query_des_idx'].to(device), test_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(test_batch['query_idx'].to(device), test_batch['query_mask'].to(device),
                                       test_batch['doc_idx'].to(device), test_batch['doc_mask'].to(device))
            
            if args.task == 'classification' or args.task == "prompt_classification":
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            elif args.task == "prompt_ranking":
                batch_score = batch_score[:, 0]
            
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict






def dev(args, model, metric, dev_loader, device,tokenizer):
    rst_dict = {}
    for dev_batch in tqdm(dev_loader, disable=args.local_rank not in [-1, 0]):
        if args.model=="t5" :
            query_id, doc_id, label = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label_id']
        else:
            query_id, doc_id, label = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label']
        with torch.no_grad():
            #if args.original_t5:
            #    batch_logits = model(                        
            #            input_ids=dev_batch['input_ids'].to(device), 
            #            attention_mask=dev_batch['attention_mask'].to(device), 
            #            labels=dev_batch["labels"].to(device),
            #            return_dict=True
             #       ).logits
             #   batch_score=batch_logits[:,0,[6136,1176]]
                #print(batch_score.shape)
            if args.model== 't5':
                batch_score,_=model(
                        input_ids=dev_batch['input_ids'].to(device), 
                        attention_mask=dev_batch['attention_mask'].to(device),
                        query_ids=dev_batch['query_ids'].to(device), 
                        doc_ids=dev_batch['doc_ids'].to(device),
                        query_attention_mask=dev_batch['query_attention_mask'].to(device),
                        doc_attention_mask=dev_batch['doc_attention_mask'].to(device),
                        labels=dev_batch['labels'].to(device)
                )
            elif args.model == 'bert':
                if args.task.startswith("prompt"):
                     batch_score, _ = model(
                        dev_batch['input_ids'].to(device),dev_batch['query_ids'].to(device),dev_batch['doc_ids'].to(device), 
                        dev_batch['mask_pos'].to(device),
                        dev_batch['input_mask'].to(device),dev_batch['query_input_mask'].to(device),dev_batch['doc_input_mask'].to(device),
                        dev_batch['segment_ids'].to(device)
                        )
                else:
                    batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                if args.task.startswith("prompt"):
                    batch_score, _ = model(
                        dev_batch['input_ids'].to(device), dev_batch['query_ids'].to(device), dev_batch['doc_ids'].to(device), 
                        dev_batch['mask_pos'].to(device), 
                        dev_batch['input_mask'].to(device), dev_batch['query_input_mask'].to(device), dev_batch['doc_input_mask'].to(device)
                        )
                else:
                    batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))

            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            
            if args.task == 'classification' or args.task == "prompt_classification":
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            elif args.task == "prompt_ranking":
                batch_score = batch_score[:, 0]
            
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict

def train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    with torch.no_grad():
        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        if args.metric.split('_')[0] == 'mrr':
            mes = metric.get_mrr(args.qrels, args.res, args.metric)
        else:
            mes = metric.get_metric(args.qrels, args.res, args.metric)
    if mes >= best_mes:
        best_mes = mes
        print('save_model...')
        if args.n_gpu > 1:
            torch.save(model.module.state_dict(), args.save)
        else:
            torch.save(model.state_dict(), args.save)
    print('initial result: ', mes)
    last_mes = mes
    for epoch in range(args.epoch):
        avg_loss = 0.0
        log_prob_ps = []
        log_prob_ns = []
        for step, train_batch in enumerate(train_loader):
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            batch_probs = F.gumbel_softmax(batch_probs, tau=args.tau)
            m = Categorical(batch_probs)
            action = m.sample()
            if action.sum().item() < 1:
                #m_scheduler.step()
                if (step+1) % args.eval_every == 0 and len(log_prob_ps) > 0:
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device)
                        om.utils.save_trec(args.res, rst_dict)
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)
                    if mes >= best_mes:
                        best_mes = mes
                        print('save_model...')
                        if args.n_gpu > 1:
                            torch.save(model.module.state_dict(), args.save)
                        else:
                            torch.save(model.state_dict(), args.save)
                    
                    print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                    avg_loss = 0.0

                    reward = mes - last_mes
                    last_mes = mes
                    if reward >= 0:
                        policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                    else:
                        policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    p_optim.step()
                    p_optim.zero_grad()

                    if args.reset:
                        state_dict = torch.load(args.save)
                        model.load_state_dict(state_dict)
                        last_mes = best_mes
                    log_prob_ps = []
                    log_prob_ns = []
                continue

            filt = action.nonzero().squeeze(-1).cpu()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device),
                                               train_batch['input_mask_pos'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device),
                                               train_batch['input_mask_neg'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device),
                                           train_batch['input_mask'].index_select(0, filt).to(device),
                                           train_batch['segment_ids'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device), train_batch['input_mask_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device), train_batch['input_mask_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device), train_batch['input_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_pos_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_ent_idx'].index_select(0, filt).to(device), train_batch['doc_pos_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_pos_des_idx'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_neg_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_ent_idx'].index_select(0, filt).to(device), train_batch['doc_neg_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_neg_des_idx'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_ent_idx'].index_select(0, filt).to(device), train_batch['doc_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_des_idx'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_idx'].index_select(0, filt).to(device), train_batch['doc_pos_mask'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_idx'].index_select(0, filt).to(device), train_batch['doc_neg_mask'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_idx'].index_select(0, filt).to(device), train_batch['doc_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            mask = action.ge(0.5)
            log_prob_p = m.log_prob(action)
            log_prob_n = m.log_prob(1-action)
            log_prob_ps.append(torch.masked_select(log_prob_p, mask))
            log_prob_ns.append(torch.masked_select(log_prob_n, mask))

            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            
            if args.n_gpu > 1:
                batch_loss = batch_loss.mean(-1)
            batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if args.n_gpu > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                avg_loss = 0.0

                reward = mes - last_mes
                last_mes = mes
                if reward >= 0:
                    policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                else:
                    policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                p_optim.step()
                p_optim.zero_grad()

                if args.reset:
                    state_dict = torch.load(args.save)
                    model.load_state_dict(state_dict)
                    last_mes = best_mes
                log_prob_ps = []
                log_prob_ns = []

def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, test_loader,device, train_sampler=None, tokenizer=None):
    best_mes = 0.0
    best_step=0
    global_step = 0 # steps that outside epoches
    force_break = False
    for epoch in range(args.epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            # print("hello?")
            if args.model=="t5":
                label = train_batch['label_id']
            else:
                label = train_batch['label']
            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext
            #if args.original_t5:
            #    with sync_context():
            #        batch_loss = model(                        
            #            input_ids=train_batch['input_ids'].to(device), 
            #            attention_mask=train_batch['attention_mask'].to(device), 
            #            labels=train_batch["labels"].to(device),
            #            return_dict=True
            #        ).loss
            if args.model == 't5':
                with sync_context():
                    batch_score,batch_loss = model(
                        input_ids=train_batch['input_ids'].to(device), 
                        attention_mask=train_batch['attention_mask'].to(device),
                        query_ids=train_batch['query_ids'].to(device), 
                        doc_ids=train_batch['doc_ids'].to(device),
                        query_attention_mask=train_batch['query_attention_mask'].to(device),
                        doc_attention_mask=train_batch['doc_attention_mask'].to(device),
                        labels=train_batch['labels'].to(device)
                        )
            elif args.model == 'bert':
                if args.task == 'ranking':
                    # sync gradients only at gradient accumulation step
                    with sync_context():
                        batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                        batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    with sync_context():
                        batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                elif args.task == "prompt_ranking":
                    with sync_context():
                        rel_and_irrel_logits_pos, masked_token_logits_pos = model(train_batch['input_ids_pos'].to(device), train_batch['mask_pos_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                        rel_and_irrel_logits_neg, masked_token_logits_neg = model(train_batch['input_ids_neg'].to(device), train_batch['mask_pos_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                        # print(rel_and_irrel_logits_pos>rel_and_irrel_logits_neg)
                        # input()
                elif args.task == "prompt_classification":
                    with sync_context():
                        batch_score, masked_token_logits = model(
                             train_batch['input_ids'].to(device), train_batch['query_ids'].to(device), train_batch['doc_ids'].to(device), 
                            train_batch['mask_pos'].to(device), 
                            train_batch['input_mask'].to(device),train_batch['query_input_mask'].to(device),train_batch['doc_input_mask'].to(device),
                            train_batch['segment_ids'].to(device)
                            )
                        # max_token_id = torch.argmax(masked_token_logits, 1).detach().cpu().tolist()  # batch_size
                        # _, topk_indices = torch.topk(masked_token_logits, 10)
                        # topk_indices = topk_indices.detach().cpu().tolist()
                        # for topk in topk_indices:
                        #     print(tokenizer.convert_ids_to_tokens(topk))
                        # print(max_token_id)
                        # print(tokenizer.convert_ids_to_tokens(max_token_id))
                        # input()
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                elif args.task == "prompt_classification":
                    with sync_context():
                        batch_score, masked_token_logits = model(
                            train_batch['input_ids'].to(device), train_batch['query_ids'].to(device), train_batch['doc_ids'].to(device), 
                            train_batch['mask_pos'].to(device), 
                            train_batch['input_mask'].to(device),train_batch['query_input_mask'].to(device),train_batch['doc_input_mask'].to(device)
                            )
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_pos_ent_idx'].to(device), train_batch['doc_pos_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_pos_des_idx'].to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_neg_wrd_idx'].to(device), train_batch['doc_neg_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_neg_ent_idx'].to(device), train_batch['doc_neg_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_neg_des_idx'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                           train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device),
                                           train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                           train_batch['doc_ent_idx'].to(device), train_batch['doc_ent_mask'].to(device),
                                           train_batch['query_des_idx'].to(device), train_batch['doc_des_idx'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    with sync_context():
                        batch_score_pos, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                                train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                        batch_score_neg, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                                train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification' or "prompt_classification":
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            
            if args.task == 'ranking':
                with sync_context():
                    if args.ranking_loss == 'margin_loss':
                        batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
                    elif args.ranking_loss == 'CE_loss':
                        batch_loss = loss_fn(torch.sigmoid(batch_score_pos-batch_score_neg),torch.ones(batch_score_neg.size()).to(device))
                    elif args.ranking_loss == 'triplet_loss':
                        logit_matrix = torch.cat([batch_score_pos.reshape([-1,1]),batch_score_neg.reshape([-1,1])], dim=1)
                        lsm = F.log_softmax(input=logit_matrix,dim=1)
                        batch_loss = torch.mean(-1.0 * lsm[:, 0])
                    elif args.ranking_loss == 'LCE_loss':
                        pass
            elif args.task == 'classification' or args.task == "prompt_classification":
                with sync_context():
                    # print(batch_score)
                    if args.original_t5:
                        pass
                    elif args.ranking_loss == "bce":
                        batch_loss = loss_fn(batch_score[:, 1], label.type(torch.FloatTensor).to(device))
                    else:
                        #print(train_batch['label'])
                        batch_loss = loss_fn(batch_score, label.to(device))
                    # print(train_batch['label'])
                    # input()
            elif args.task == "prompt_ranking":
                with sync_context():
                    loss_rel = loss_fn(rel_and_irrel_logits_pos[:, 0], rel_and_irrel_logits_neg[:, 0], torch.ones(rel_and_irrel_logits_pos[:, 0].size()).to(device))
                    loss_irrel = loss_fn(rel_and_irrel_logits_neg[:, 1], rel_and_irrel_logits_pos[:, 1], torch.ones(rel_and_irrel_logits_neg[:, 1].size()).to(device))
                    batch_loss = loss_rel + loss_irrel
            else:
                raise ValueError('Task must be `ranking` or `classification`.')

            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            with sync_context():
                batch_loss.backward()
            # if args.local_rank != -1:
            #     if (step+1) % args.gradient_accumulation_steps == 0:
            #         batch_loss.backward()
            #     else:
            #         with model.no_sync():
            #             batch_loss.backward()
            # else:
            #     batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                m_optim.step()
                if m_scheduler is not None:
                    m_scheduler.step()
                m_optim.zero_grad()
                # global_step += 1
                # print("step")

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    dist.barrier()
                    if args.local_rank in [-1, 0]:
                        logger.info("training gpu {}:,  global step: {}, local step: {}, loss: {}".format(args.local_rank,global_step+1, step+1, avg_loss/args.logging_step))
                        if args.tb is not None:
                            args.tb.add_scalar("loss", avg_loss/args.logging_step, global_step + 1)
                            args.tb.add_scalar("epochs", epoch + 1, global_step + 1)
                        
                    dist.barrier()
                    avg_loss = 0.0 

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):                
                    model.eval()
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device,tokenizer=tokenizer)
                    model.train()

                    if args.local_rank != -1:
                        # distributed mode, save dicts and merge
                        om.utils.save_trec(args.res + "_rank_{:03}".format(args.local_rank), rst_dict)
                        dist.barrier()
                        # if is_first_worker():
                        if args.local_rank in [-1,0]:
                            merge_resfile(args.res + "_rank_*", args.res)

                    else:
                        om.utils.save_trec(args.res, rst_dict)
                        
                    # if is_first_worker():
                    if args.local_rank in [-1,0]:
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)

                        if args.tb is not None:
                            args.tb.add_scalar("dev", mes, global_step + 1)

                        #logger.info('save_model at step {}'.format(global_step+1))
                        if not os.path.exists(args.save):
                                os.makedirs(args.save)

                        if mes>best_mes:
                            best_mes=mes
                            #best_step=global_step+1
                            ls=os.listdir(args.save)
                            for i in ls:
                                item_path=os.path.join(args.save,i)
                                #print("remove {}".format(item_path))
                                logger.info('remove_model at step {}'.format(global_step+1))
                                logger.info('save model')
                                os.remove(item_path)
                            #if args.soft_prompt:
                                #model.module.save_prompts(args.save+"_step-{}.pickle".format(global_step+1))
                            if hasattr(model, "module"):
                                torch.save(model.module.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                            else:
                                torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                        # if args.n_gpu > 1:
                        #     torch.save(model.module.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                        # else:
                        #     torch.save(model.state_dict(), args.save + "_step-{}.bin".format(global_step+1))
                        logger.info("global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))
                
                global_step += 1

                if args.max_steps is not None and global_step == args.max_steps:
                    force_break = True
                    break
                
        if force_break:
            break
    if not args.no_test:
        if args.local_rank != -1:
            dist.barrier()
            logger.info("load best checkpoint....")
            dist.barrier()
            for file in os.listdir(args.save):
                checkpoint=os.path.join(args.save,file)
                state=torch.load(checkpoint,map_location=device)
                if args.model == 'bert':
                    st = {}
                    for k in state:
                        if k.startswith('bert'):
                            st['_model'+k[len('bert'):]] = state[k]
                        elif k.startswith('classifier'):
                            st['_dense'+k[len('classifier'):]] = state[k]
                        else:
                            st[k] = state[k]
                    model.module.load_state_dict(st)
                else:
                    model.module.load_state_dict(state)
            dist.barrier()
            logger.info("doing inference.... at gpu:{}".format(args.local_rank))
            model.eval()
            with torch.no_grad():
                rst_dict = test(args, model, metric, test_loader, device,tokenizer=tokenizer)
            om.utils.save_trec(args.test_res + "_rank_{:03}".format(args.local_rank), rst_dict)
            logger.info("inference finished...at gpu:{}".format(args.local_rank))
            dist.barrier()
            # if is_first_worker():
            if args.local_rank in [-1,0]:
                merge_resfile(args.test_res + "_rank_*", args.test_res)
            dist.barrier()

    return 
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-ranking_loss', type=str, default='margin_loss')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-seed', type=int, default=13)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-test', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-warmed_checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-test_res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float,help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)
    parser.add_argument("--log_dir", type=str)

    parser.add_argument("--template", type=str, default="[SP0] <q> <mask> <d>")
    parser.add_argument("--pos_word", type=str, default=" relevant")
    parser.add_argument("--neg_word", type=str, default=" irrelevant")
    parser.add_argument("--soft_prompt", action="store_true")
    parser.add_argument("--infix",type=str,default=None)
    parser.add_argument("--suffix",type=str,default=None)
    parser.add_argument("--prefix",type=str,default=None)
    parser.add_argument("--soft_sentence",type=str,default=None)
    parser.add_argument("--original_t5", action="store_true")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--no_test", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    set_dist_args(args) # get local cpu/gpu device
    #print(args.template)
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
        args.tb = writer
    else:
        args.tb = None

    args.model = args.model.lower()
    tokenizer = None

    if args.task.startswith("prompt"):
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        pos_word_id = tokenizer(args.pos_word, add_special_tokens=False)["input_ids"]
        neg_word_id = tokenizer(args.neg_word, add_special_tokens=False)["input_ids"]
        print(pos_word_id, neg_word_id)
        if len(neg_word_id) > 1 or len(pos_word_id) > 1:
            raise ValueError("Label words longer than 1 after tokenization")
        pos_word_id = pos_word_id[0]
        neg_word_id = neg_word_id[0]
        # tokenizer.add_tokens(["[SP1]", "[SP2]", "[SP3]", "[SP4]"], special_tokens=True)  # For continuous prompt

    if args.model=="t5":
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        logger.info('reading training data...')
        train_set = om.data.datasets.t5Dataset(
                dataset=args.train,
                tokenizer=tokenizer,
                max_input=args.max_input,
                template=args.template
            )
        logger.info('reading dev data...')
        dev_set = om.data.datasets.t5Dataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                max_input=args.max_input,
                template=args.template
            )
        logger.info("reading test data...")
        test_set = om.data.datasets.t5Dataset(
                dataset=args.test,
                tokenizer=tokenizer,
                max_input=args.max_input,
                template=args.template
            )
    elif args.model == 'bert':
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        logger.info('reading training data...')
        if args.maxp:
            train_set = om.data.datasets.BertMaxPDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            train_set = om.data.datasets.BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task,
                template=args.template
            )
        logger.info('reading dev data...')
        if args.maxp:
            dev_set = om.data.datasets.BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            dev_set = om.data.datasets.BertDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
            )
        logger.info('reading test data...')
        if args.maxp:
            test_set = om.data.datasets.BertMaxPDataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            test_set = om.data.datasets.BertDataset(
                dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
            )
    elif args.model == 'roberta':
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = om.data.datasets.RobertaDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
        )
        print('reading dev data...')
        dev_set = om.data.datasets.RobertaDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
        )
        print('reading test data...')
        test_set = om.data.datasets.RobertaDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.EDRMDataset(
            dataset=args.train,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.EDRMDataset(
            dataset=args.dev,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading test data...')
        test_set = om.data.datasets.EDRMDataset(
            dataset=args.test,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading test data...')
        test_set = om.data.datasets.Dataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

    if args.local_rank != -1:
        # train_sampler = DistributedSampler(train_set, args.world_size, args.local_rank)
        train_sampler = DistributedSampler(train_set)
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            sampler=train_sampler
        )
        #dev_sampler = DistributedSampler(dev_set)
        dev_sampler = DistributedEvalSampler(dev_set)
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
            shuffle=False,
            num_workers=16,
            sampler=dev_sampler
        )
        test_sampler = DistributedEvalSampler(test_set)
        test_loader = om.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
            shuffle=False,
            num_workers=16,
            sampler=test_sampler
        )
        dist.barrier()

    else:
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.dev_eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        test_loader = om.data.DataLoader(
            dataset=test_set,
            batch_size=args.dev_eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        train_sampler = None

    if args.model == "t5":
        model = om.models.t5(args.pretrain,args.original_t5,args.soft_prompt,args.prefix,args.infix,args.suffix) 
    elif args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        else:
            if args.task.startswith("prompt"):
                model = om.models.BertPrompt(
                    pretrained=args.pretrain,
                    mode=args.mode,
                    task=args.task,
                    pos_word_id=pos_word_id,
                    neg_word_id=neg_word_id,
                    soft_prompt=args.soft_prompt,
                    prefix=args.prefix,
                    suffix=args.suffix
                )
                model._model.resize_token_embeddings(len(tokenizer))
            else:
                model = om.models.Bert(
                    pretrained=args.pretrain,
                    mode=args.mode,
                    task=args.task
                )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
    elif args.model == 'edrm':
        model = om.models.EDRM(
            wrd_vocab_size=tokenizer.get_vocab_size(),
            ent_vocab_size=ent_tokenizer.get_vocab_size(),
            wrd_embed_dim=tokenizer.get_embed_dim(),
            ent_embed_dim=128,
            max_des_len=20,
            max_ent_num=3,
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            wrd_embed_matrix=tokenizer.get_embed_matrix(),
            ent_embed_matrix=None,
            task=args.task
        )
    elif args.model == 'tk':
        model = om.models.TK(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            head_num=10,
            hidden_dim=100,
            layer_num=2,
            kernel_num=args.n_kernels,
            dropout=0.0,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking' or args.task == "prompt_ranking":
            if args.ranking_loss == 'margin_loss':
                loss_fn = nn.MarginRankingLoss(margin=1)
            elif args.ranking_loss == 'CE_loss':
                loss_fn = nn.BCELoss()
            elif args.ranking_loss == 'triplet_loss':
                loss_fn = nn.BCELoss() # dummpy loss for occupation
                # loss_fn = F.log_softmax(dim=1)
            elif args.ranking_loss == 'LCE_loss':
                print("LCE loss TODO")
                # nn.CrossEntropyLoss()

        elif args.task == 'classification' or args.task == "prompt_classification":
            if args.ranking_loss == "bce":
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')


    model.to(device)
    ckpt=args.warmed_checkpoint
    if ckpt is not None and args.model=="t5":
        st = torch.load(ckpt,map_location=device)
        if args.soft_prompt:
            st['suffix_soft_embedding_layer.weight']=model.state_dict()['suffix_soft_embedding_layer.weight'].clone().detach()
            st['infix_soft_embedding_layer.weight']=model.state_dict()['infix_soft_embedding_layer.weight'].clone().detach()
            st['prefix_soft_embedding_layer.weight']=model.state_dict()['prefix_soft_embedding_layer.weight'].clone().detach()
        model.load_state_dict(st)
    logger.info("load mnli-pretrained checkpoint at gpu:{}...".format(args.local_rank))
    dist.barrier()
    if args.reinfoselect:
        policy.to(device)
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
    #print(model.prefix_soft_embedding_layer.weight.data.requires_grad)
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == "adafactor":
        m_optim = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
        # print(m_optim.param_groups)
    # from IPython import embed
    # embed()
    #print(len(m_optim.param_groups[0]))
    #for p in m_optim.param_groups[0]['params']:
        #print(p.shape,p.requires_grad)
    #print(len(m_optim.param_groups[0]['params'])) 
    if args.optimizer.lower() == "adafactor":
        m_scheduler = None
    elif args.local_rank == -1:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size if args.max_steps is None else args.max_steps)
    else:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.world_size*args.gradient_accumulation_steps) if args.max_steps is None else args.max_steps)
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)

    if m_optim is not None:
        optimizer_to(m_optim,device)
    

    metric = om.metrics.Metric()

    logger.info(args)
    if args.reinfoselect:
        train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader,test_loader, device)
    else:
        train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, test_loader,device, train_sampler=train_sampler, tokenizer=tokenizer)
    # print("outside train. {}".format(args.local_rank))
    if args.local_rank != -1:
        dist.barrier()
    return args

if __name__ == "__main__":
    main()
    os._exit(0)

