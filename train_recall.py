import argparse
import json
import os
import time
from pathlib import Path
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from dataset import MyDataset
from utils import (save_item_id,get_cosine_schedule_with_warmup_custom,InfoNCELoss,MRR)
from recall.DIFSR_NSToken import DIFSRwithNSTokens
from dotenv import load_dotenv
import contextlib


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_len', default=8, type=int)
    parser.add_argument('--num_epochs', default=28, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    # InfoNCE specific parameters
    parser.add_argument('--temperature', default=0.03, type=float, help='Temperature parameter for InfoNCE loss')
    
    # Optimizer and scheduler
    parser.add_argument('--weight_decay', default=0, type=float, help="Weight decay for AdamW optimizer")
    parser.add_argument('--warmup_steps', default=int(743*2), type=int, help="Number of steps for learning rate warmup")
    parser.add_argument('--clip_norm', default=1.0, type=float, help="Max norm for gradient clipping")

    # Baseline Model construction
    parser.add_argument('--embedding_units', default=128, type=int)
    parser.add_argument('--hidden_units', default=250, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_buckets', default=128, type=int)
    parser.add_argument('--norm_first', default=True)

    #Ohters
    parser.add_argument('--l2_emb', default=0.0000, type=float)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--data_path', default='data')
    parser.add_argument('--reuslt_path', default='result')

    #recall
    parser.add_argument('--topk', default=5,type=int)

    #仅test生成
    parser.add_argument('--inference_only', action='store_true')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.manual_seed(3407) #3407
    random.seed(3407)
    args = get_args()

    #加载数据集
    dataset = MyDataset(args,mode="train")
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=MyDataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=MyDataset.collate_fn
    )
    # 假设 valid_loader 已经按你的代码定义好了
    validation_ground_truth = {}

    print("Building validation ground truth dictionary...")
    # 迭代验证集数据加载器
    # 使用 torch.no_grad() 不是必须的，但这是个好习惯，可以节省内存
    with torch.no_grad():
        for batch_data in valid_loader:
            # 1. 从 batch 中获取 user_id 和 pos_id
            #    batch_data['user_id'] 是形状为 [B] 的 Tensor
            #    batch_data['pos_id'] 也是形状为 [B] 的 Tensor
            
            # 2. 将 Tensors 转换为 Python 列表，方便遍历
            user_ids_list = batch_data['user_id'].cpu().tolist()
            pos_ids_list = batch_data['pos_id'].cpu().tolist()
            
            # 3. 遍历 batch 内的每一个样本，并存入字典
            for uid, true_next_id in zip(user_ids_list, pos_ids_list):
                # 确保 uid 不是0 (如果是 padding) 并且 true_next_id 也不是 0 (如果是占位符)
                # 在验证集中，pos_id 应该就是那个真实 ID
                if true_next_id != 0:
                    validation_ground_truth[uid] = true_next_id

    print(f"Ground truth dictionary built. Total validation users: {len(validation_ground_truth)}")

    #创建模型
    model = DIFSRwithNSTokens(args,max_time_diff=dataset.share_norm['time_diff'][1]).to(args.device)
    #初始化
    for name, param in model.named_parameters():
        if param.dim() > 1:
            try:
                torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
            except Exception:
                pass
    #词表0初始化:同起点让模型更好看出区别
    #model.wordscate_embedding.weight.data[:, :] = 0
    #model.articlecate_embedding.weight.data[:, :] = 0
    #for key in ["user_click_env",'user_click_dev','user_click_os',"user_click_region","user_click_refer"]:
    #    model.user_sparse_emb[key].weight.data[:, :] = 0

    #损失函数
    infonce_criterion = InfoNCELoss(temperature=args.temperature,reduction="mean").to(args.device)
    
    #优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    # 计算总训练步数
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup_custom(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        start_lr_factor=0.1,
        end_lr_factor=0
    )

    #梯度Scaler
    scaler = torch.amp.GradScaler('cuda')

    print("=" * 60)
    print("开始模型训练...")
    print("=" * 60)

    #环境变量
    load_dotenv() 
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')

    #epoch信息
    epoch_start_idx = 1
    T = 0.0
    t0 = time.time()
    global_step = 0


    #'''
    print("Start training with InfoNCE loss")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            #混合精度
            with torch.amp.autocast('cuda'):
            #with contextlib.nullcontext():
                batch_data = {k: v.to(args.device) for k, v in  batch_data.items()}
                #for k,v in batch_data.items():
                #    print(f"{k}",v[0])

                optimizer.zero_grad()

                #获取最终表征(已正则化)
                final_norm ,token_mask_2,batch_data  = model.emb2emb(batch_data)
                #正负样本进行标准化
                pos_norm = F.normalize(batch_data["pos_seq"],p=2,dim=-1)
                neg_norm = F.normalize(batch_data["neg_seq"],p=2,dim=-1)

                #计算损失
                loss_mask = (token_mask_2 != 0).to(args.device)
                loss_dict_next = infonce_criterion(final_norm, pos_norm,neg_norm, loss_mask)
                loss = loss_dict_next['loss']
                #反向传播
                '''
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                if torch.isfinite(total_grad_norm):
                    writer.add_scalar('Gradient/Total_Norm_before_clip', total_grad_norm.item(), global_step)
                else:
                    # 如果梯度是 inf 或 nan (在 float32 下也可能发生)
                    writer.add_scalar('Gradient/Total_Norm_before_clip', -1.0, global_step)
                optimizer.step()
                scheduler.step() 
                '''
                # 1. 计算放大的梯度
                scaler.scale(loss).backward()
                # 2. [关键] 在裁剪前，手动将梯度解缩放 (unscale)
                #    scaler 会检查是否有 inf/NaN，然后将 model.parameters() 的 .grad 属性
                #    就地（in-place）除以 scale factor
                scaler.unscale_(optimizer)
                # 3. 现在，.grad 已经是正常的梯度了，可以安全地裁剪
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                if torch.isfinite(total_grad_norm):
                    writer.add_scalar('Gradient/Total_Norm_before_clip', total_grad_norm.item(), global_step)
                else:
                    # 如果梯度是 inf 或 nan，记录一个特殊值 (比如 -1) 来标记它
                    writer.add_scalar('Gradient/Total_Norm_before_clip', -1.0, global_step)

                # 4. Scaler 执行 step (它内部知道梯度已经 unscale 过了)
                scaler.step(optimizer)
                # 5. 更新 scale factor
                scaler.update()
                # 6. 更新学习率
                scheduler.step()  
                

            #记录
            writer.add_scalar('Loss/train_next_infonce', loss.item(), global_step)
            writer.add_scalar('Model/pos_logits_step1', loss_dict_next['pos_logits_mean'].item(), global_step)
            writer.add_scalar('Model/neg_logits_step1', loss_dict_next['neg_logits_mean'].item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)  #记录学习率
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time(),
                 'loss_next': loss.item()} #'loss_rand': loss_random_action.item() if isinstance(loss_random_action, torch.Tensor) else 0}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            global_step += 1

        #评估模式：使dropout这种进入测试模式
        model.eval()
        # 在验证模式下，不需要计算梯度
        topk = args.topk
        S_recall_df = pd.DataFrame()
        NS_recall_df = pd.DataFrame()  
        # 初始化用于累加验证集损失
        total_val_loss = 0.0     
        #faiss库在dataset里建立         
        faiss_searcher = dataset.faiss_index
        faiss_id_map = dataset.faiss_id_map        
        with torch.no_grad(): 
            for step, batch_data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                batch_data = {k: v.to(args.device) for k, v in  batch_data.items()}

                #获取最终表征(已正则化)
                final_norm ,token_mask_2,batch_data  = model.emb2emb(batch_data)
                #正负样本进行标准化
                pos_norm = F.normalize(batch_data["pos_seq"],p=2,dim=-1)
                neg_norm = F.normalize(batch_data["neg_seq"],p=2,dim=-1)
                #计算损失
                loss_mask = (token_mask_2 != 0).to(args.device)
                loss_dict_next = infonce_criterion(final_norm, pos_norm,neg_norm, loss_mask)
                loss = loss_dict_next['loss']
                total_val_loss += loss.item()

                # --- 2. 提取 S_emb 和 NS_emb 用于召回 ---
                # final_seq_emb 形状: (B, L, H)
                
                # S_emb: 最后一个 "1" token 的表征，位于 -2 位置
                S_emb_recall = final_norm[:, -2, :] # (B, H)
                
                # NS_emb: 最后一个 "2" token 的表征，位于 -1 位置
                NS_emb_recall = final_norm[:, -1, :] # (B, H)
                
                user_ids = batch_data['user_id'] # (B,)
                
                # --- 3. 调用 save_item_id 进行召回并累加 ---
                S_recall_df, NS_recall_df = save_item_id(
                    faiss_searcher, 
                    faiss_id_map,
                    S_emb_recall, 
                    NS_emb_recall, 
                    user_ids,
                    S_recall_df, 
                    NS_recall_df, 
                    topk=topk
                )                           
            # --- 循环结束，开始计算指标 ---
            
            # 1. 计算平均验证损失
            avg_val_loss = total_val_loss / len(valid_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # 2. 计算 MRR
            print(f"Calculating MRR@{topk} for {len(S_recall_df)} users...")
            S_recall_df_with_metrics = MRR(S_recall_df, validation_ground_truth, topk=topk)
            NS_recall_df_with_metrics = MRR(NS_recall_df, validation_ground_truth, topk=topk)
            
            # 3. 计算并打印最终的平均 MRR
            s_mean_mrr = S_recall_df_with_metrics['MRR'].mean()
            ns_mean_mrr = NS_recall_df_with_metrics['MRR'].mean()
            print(f"epoch{epoch} :S_recall_MRR:{s_mean_mrr}")
            print(f"epoch{epoch} :NS_recall_MRR:{ns_mean_mrr}")

            writer.add_scalar('Valid/pos_logits_step1', loss_dict_next['pos_logits_mean'].item(), global_step)
            writer.add_scalar('Valid/neg_logits_step1', loss_dict_next['neg_logits_mean'].item(), global_step)
            writer.add_scalar('Valid/S_recall_MRR',  s_mean_mrr, global_step)
            writer.add_scalar('Valid/NS_recall_MRR', ns_mean_mrr, global_step)
        # 保存模型
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step},S_Recall_MRR={s_mean_mrr :.4f},NS_Recall_MRR={ns_mean_mrr :.4f}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        #'''

    #测试阶段：生成测试集的召回结果：（1）导出两份csv：s_recall.csv 和 ns_recall.csv(均不含"MRR"列) （2）两份召回结果拼在一起的：recall.csv(不含MRR列)
    with torch.no_grad():
        #人为选择。
        #s_mean_mrr=
        #ns_mean_mrr=
        #global_step = 
        print("\n" + "="*20 + " Starting Test Phase " + "="*20)
        
        # --- 1. 加载最佳模型 ---
        ckpt_dir_name = f"global_step{global_step},S_Recall_MRR={s_mean_mrr :.4f},NS_Recall_MRR={ns_mean_mrr :.4f}"
        load_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), ckpt_dir_name)
        model_path = load_dir / "model.pt"
        
        if os.path.exists(model_path):
            print(f"Loading best model from: {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"Warning: Model checkpoint not found at {model_path}. Using current model state.")

        model.eval()
        
        # --- 2. 准备输出目录和召回载体 ---
        output_folder = Path(args.reuslt_path) 
        output_folder.mkdir(parents=True, exist_ok=True)
        
        topk = args.topk # 召回数量
        S_recall_df = pd.DataFrame()
        NS_recall_df = pd.DataFrame()

        # 假设 test_dataset, test_loader 已经定义
        dataset.__init__(args,mode="test")
        faiss_searcher = dataset.faiss_index
        faiss_id_map = dataset.faiss_id_map
        test_loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=MyDataset.collate_fn)

        # --- 3. 遍历测试集进行召回---
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Generating Test Recall"):
                
                batch_data = {k: v.to(args.device) for k, v in batch.items()}
                #final_seq_emb, _, _ = model.emb2emb(batch_data)
                #S_emb_recall = final_seq_emb[:, -2, :]
                #NS_emb_recall = final_seq_emb[:, -1, :]

                S_emb_recall,NS_emb_recall = model.predict(batch_data)

                user_ids = batch_data['user_id']
                
                S_recall_df, NS_recall_df = save_item_id(
                    faiss_searcher, 
                    faiss_id_map,
                    S_emb_recall, 
                    NS_emb_recall, 
                    user_ids,
                    S_recall_df, 
                    NS_recall_df, 
                    topk=topk
                )

        # --- 4. 保存召回结果
        
        # 4.1 定义保存路径
        s_recall_path = output_folder / "s_recall.csv"
        ns_recall_path = output_folder / "ns_recall.csv"
        combined_recall_path = output_folder / "recall.csv"

        # 4.2 (1) 导出 S_recall.csv 和 NS_recall.csv (宽格式)
        print(f"Saving S-Token recall to {s_recall_path}...")
        S_recall_df.sort_values(by='user_id').to_csv(s_recall_path, index=False)
        
        print(f"Saving NS-Token recall to {ns_recall_path}...")
        NS_recall_df.sort_values(by='user_id').to_csv(ns_recall_path, index=False)

        # 4.3 (2) 合并 S 和 NS 召回结果 (宽格式拼接)
        print(f"Generating combined recall file at {combined_recall_path}...")
        
        # 步骤 A: 重命名 NS_recall_df 的列 (除了 'user_id')
        rename_map = {
            f'article_{i+1}': f'article_{i+1}_ns'
            for i in range(topk)
        }
        NS_recall_df_renamed = NS_recall_df.rename(columns=rename_map)

        # 步骤 B: 按照 user_id 合并
        # 假设 S 和 NS 召回的用户集是完全一致的 (使用 'inner' join)
        combined_wide_df = pd.merge(
            S_recall_df, 
            NS_recall_df_renamed, 
            on='user_id', 
            how='inner'
        )

        # 步骤 C: 保存合并后的宽格式
        combined_wide_df.sort_values(by='user_id').to_csv(combined_recall_path, index=False)
        
        print(f"Test recall generation finished. Results saved in {output_folder}")

    #结束    
    print("Done")
    writer.close()
    log_file.close()