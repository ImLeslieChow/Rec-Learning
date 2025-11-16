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
from dataset_rank import MyDataset
from utils import (get_cosine_schedule_with_warmup_custom)
from rank.DIFSR_NSToken import DIFSRwithNSTokens
from dotenv import load_dotenv


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--max_len', default=18, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    # InfoNCE specific parameters
    parser.add_argument('--temperature', default=0.03, type=float, help='Temperature parameter for InfoNCE loss')
    
    # Optimizer and scheduler
    parser.add_argument('--weight_decay', default=0, type=float, help="Weight decay for AdamW optimizer")
    parser.add_argument('--warmup_steps', default=500, type=int, help="Number of steps for learning rate warmup")
    parser.add_argument('--clip_norm', default=1.0, type=float, help="Max norm for gradient clipping")

    # Baseline Model construction
    parser.add_argument('--embedding_units', default=128, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_buckets', default=128, type=int)
    parser.add_argument('--norm_first', default=False)

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
    '''
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=MyDataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=MyDataset.collate_fn
    )
    '''
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=MyDataset.collate_fn
    )

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
    criterion = torch.nn.BCEWithLogitsLoss()
    
    #优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    # 计算总训练步数
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup_custom(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        start_lr_factor=0.1,
        end_lr_factor=0.1
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
            #epoch初始化
            total_label_0 = 0
            total_label_1 = 0
            # 初始化指标累加器
            total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
            total_correct, total_samples = 0, 0

            #混合精度
            with torch.amp.autocast('cuda'):
                batch_data = {k: v.to(args.device) for k, v in  batch_data.items()}
                #for k,v in batch_data.items():
                #    print(f"{k}",v[0])

                optimizer.zero_grad()

                #获取logits
                logits = model.emb2emb(batch_data)

                #获取labels
                labels = batch_data['Label']
                labels_float = labels.to(logits.dtype).unsqueeze(1)
                counts = torch.bincount(labels, minlength=2)
                total_label_0 += counts[0].item()
                total_label_1 += counts[1].item()
                #计算损失
                loss= criterion(logits, labels_float)

                #添加准确率、召回率等指标
                preds = (torch.sigmoid(logits) > 0.5).long()
                # 2. 准备标签 [B] -> [B, 1]
                labels_long = labels.unsqueeze(1)

                # 3. 累积总数
                total_samples += labels.size(0)
                total_correct += (preds == labels_long).sum().item()

                # 4. 累积 TP, TN, FP, FN
                total_tp += ((preds == 1) & (labels_long == 1)).sum().item()
                total_tn += ((preds == 0) & (labels_long == 0)).sum().item()
                total_fp += ((preds == 1) & (labels_long == 0)).sum().item()
                total_fn += ((preds == 0) & (labels_long == 1)).sum().item()

                #反向传播
                '''
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()   
                '''
                # 1. 计算放大的梯度
                scaler.scale(loss).backward()
                # 2. [关键] 在裁剪前，手动将梯度解缩放 (unscale)
                #    scaler 会检查是否有 inf/NaN，然后将 model.parameters() 的 .grad 属性
                #    就地（in-place）除以 scale factor
                scaler.unscale_(optimizer)
                # 3. 现在，.grad 已经是正常的梯度了，可以安全地裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                # 4. Scaler 执行 step (它内部知道梯度已经 unscale 过了)
                scaler.step(optimizer)
                # 5. 更新 scale factor
                scaler.update()
                # 6. 更新学习率
                scheduler.step()


            #记录
            writer.add_scalar('Loss/train_next_infonce', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)  #记录学习率
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()} #'loss_rand': loss_random_action.item() if isinstance(loss_random_action, torch.Tensor) else 0}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            global_step += 1

        # 打印标签计数
        print(f"\nEpoch finished. Total Label 0 generated: {total_label_0}")
        print(f"Epoch finished. Total Label 1 generated: {total_label_1}")

        # --- 计算并打印整个 Epoch 的指标 ---
        epsilon = 1e-8 # 防止除零
        epoch_accuracy = total_correct / (total_samples + epsilon)
        epoch_precision = total_tp / (total_tp + total_fp + epsilon)
        epoch_recall = total_tp / (total_tp + total_fn + epsilon)
        epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall + epsilon)

        # 打印指标
        print("--- Epoch Metrics ---")
        print(f"Accuracy:  {epoch_accuracy:.4f}")
        print(f"Precision: {epoch_precision:.4f} (TP: {total_tp}, FP: {total_fp})")
        print(f"Recall:    {epoch_recall:.4f} (TP: {total_tp}, FN: {total_fn})")
        print(f"F1-Score:  {epoch_f1:.4f}")
        print("---------------------")
        writer.add_scalar('Metrics/Accuracy', epoch_accuracy, global_step)
        writer.add_scalar('Metrics/Precision', epoch_precision, global_step)
        writer.add_scalar('Metrics/Recall', epoch_recall, global_step)
        writer.add_scalar('Metrics/F1-Score', epoch_f1, global_step)

        #保存模型
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step},Accuracy{epoch_accuracy:4f},Precision{epoch_precision:.4f},Recall{epoch_recall:.4f},F1-Score{epoch_f1:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

        #无评估模式：全量
        # 在 epoch 循环结束后，打印总数
        print("="*30)
        print(f"Epoch finished. Total Label 0 generated: {total_label_0}")
        print(f"Epoch finished. Total Label 1 generated: {total_label_1}")
        print(f"Total samples: {total_label_0 + total_label_1}")
        print("="*30)
        #'''

    #测试阶段：生成测试集的排序结果：读取reuslt中的recall.csv进行排序
    with torch.no_grad():
        #人为选择。
        #global_step = 
        #epoch_accuracy = 
        #epoch_precision = 
        #epoch_recall = 
        #epoch_f1 = 
        print("\n" + "="*20 + " Starting Test Phase " + "="*20)
        
        # --- 1. 加载最佳模型 ---
        ckpt_dir_name = f"global_step{global_step},Accuracy{epoch_accuracy:4f},Precision{epoch_precision:.4f},Recall{epoch_recall:.4f},F1-Score{epoch_f1:.4f}"
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
        
        # 最终要提交的 topk (e.g., 50)
        topk_final = args.topk 
        
        recall_path = Path(args.reuslt_path,"recall.csv")
        recall_df = pd.read_csv(recall_path)
        
        uid_nums = recall_df.shape[0] # 一行一个用户
        # 获取召回的列名，例如 ['article_1', 'article_2', ...]
        recall_cols = [col for col in recall_df.columns if col.startswith('article_')]
        recall_nums = len(recall_cols) # 召回物品的数量
        print(f"Loaded {uid_nums} users with {recall_nums} candidates each for reranking.")


        dataset.__init__(args,mode="test")
    
        # 1. 对同批次按召回顺序进行打分
        # raw_rank 用于存储模型输出的 logits 分数
        raw_rank = pd.DataFrame(columns=['user_id'] + recall_cols)
        raw_rank['user_id'] = recall_df['user_id']
        
        with torch.no_grad():
            # [修复 Bug] 步长应该是 args.batch_size
            for i in tqdm(range(0, uid_nums, args.batch_size), desc="Reranking"):
                start = i
                end   = min(i + args.batch_size, uid_nums)
                
                # 截取当前批次的用户和他们的召回列表
                batch_df_slice = recall_df.iloc[start:end]
                batch_ids = batch_df_slice['user_id'].values # [B,]

                # 用于存储当前批次所有分数 [B, recall_nums]
                batch_scores_np = np.zeros((len(batch_ids), recall_nums))
                
                # 给第k个召回候选进行打分
                # k 是 0-based 索引, col_idx 是 1-based (k+1)
                for k in range(recall_nums):
                    col_name = recall_cols[k] # e.g., "article_1"
                    batch_candidate = batch_df_slice[col_name].values # [B,]

                    # [关键] 调用 dataset.rank_batch
                    batch = dataset.rank_batch(batch_ids, batch_candidate)
                    
                    if batch is None: # 如果 batch 为空（例如 uid 找不到）
                        batch_scores_np[:, k] = -np.inf # 设为极小值
                        continue
                        
                    batch_data = {key: v.to(args.device) for key, v in batch.items()}

                    # [关键] 获取 logits
                    # 假设 model(batch_data) 直接返回 [B, 1] 的 logits
                    logits = model.emb2emb(batch_data) 

                    # [修复 Bug] 将 logits 存入 numpy 数组
                    batch_scores_np[:, k] = logits.detach().cpu().numpy().squeeze()
                
                # [修复 Bug] 批量写入分数，比 .loc 高效
                raw_rank.iloc[start:end, 1:] = batch_scores_np # 1: 是因为第0列是 user_id

        # --- 2. 对 raw_rank 按分数排序，导出最终的 topk ---
        print("Sorting results...")
        
        # 转换 DataFrame 为 NumPy 数组以便快速操作
        # 1. 原始召回的 item_id 矩阵
        candidate_ids_np = recall_df[recall_cols].values # (N, recall_nums)
        # 2. 对应的分数矩阵
        scores_np = raw_rank[recall_cols].values        # (N, recall_nums)
        
        # 3. [核心] 得到按分数降序排序的 *索引*
        #    np.argsort(-scores_np) 会返回每一行中，分数从大到小对应的 *列索引*
        sorted_indices = np.argsort(-scores_np, axis=1) # (N, recall_nums)
        
        # 4. 使用这些索引去 "重排" 原始的 item_id 矩阵
        sorted_candidate_ids = np.take_along_axis(candidate_ids_np, sorted_indices, axis=1)
        
        # 5. 截取最终需要的 topk_final
        #final_topk_ids = sorted_candidate_ids[:, :topk_final] # (N, topk_final)
        # 5. [新增] 逐行去重并截取 topk_final
        final_topk_lists = []
        # 遍历 N 个用户 (N 行)
        for i in range(len(sorted_candidate_ids)):
            user_row = sorted_candidate_ids[i] # (recall_nums,)
            
            user_topk_dedup = []
            seen_items = set()
            
            for item_id in user_row:
                if item_id not in seen_items:
                    user_topk_dedup.append(item_id)
                    seen_items.add(item_id)
                
                # 当我们收集到足够的 item 时，立即停止
                if len(user_topk_dedup) == topk_final:
                    break
            
            # [兜底] 如果去重后不足 topk_final，用 0 填充
            padding_needed = topk_final - len(user_topk_dedup)
            if padding_needed > 0:
                user_topk_dedup.extend([0] * padding_needed) # 假设 0 是一个安全的 padding 值
                
            final_topk_lists.append(user_topk_dedup)

        # 6. 将 Python 列表转回 NumPy 数组
        final_topk_ids = np.array(final_topk_lists) # (N, topk_final)   
             
        # --- 3. 创建并保存最终的提交文件 ---
        rank_df = pd.DataFrame(final_topk_ids)
        rank_df.insert(0, 'user_id', recall_df['user_id'])
        
        # 重命名列
        rank_cols = ['user_id'] + [f'article_{i+1}' for i in range(topk_final)]
        rank_df.columns = rank_cols
        
        submission_path = output_folder / "submission_ranked.csv"
        rank_df.to_csv(submission_path, index=False)
        
        # (可选) 保存原始分数
        raw_rank.to_csv(output_folder / "raw_scores.csv", index=False)
        
        print(f"Reranking finished. Final submission saved to: {submission_path}")

    #结束    
    print("Done")
    writer.close()
    log_file.close()