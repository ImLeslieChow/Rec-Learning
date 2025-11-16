import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import torch

def save_item_id(searcher, id_map, S_emb, NS_emb, user_id, S_dataframe, NS_dataframe, topk=5):
    '''
    使用faiss库生成召回item id
    
    输入：
        searcher：已建立的faiss库对象
        id_map: faiss库中id的映射表 (NumPy array)
        S_emb、NS_emb: [B, D] - PyTorch Tensors
        user_id : [B,] - PyTorch Tensor
        S_dataframe: 存储S_token召回的item
        NS_dataframe：储存NS_token召回的item
    输出：
        将该批次的召回结果拼接到已有的pd.dataframe中(添加行)
    '''
    
    # --- 1. 处理 S_token 的召回结果 ---
    S_emb_np = S_emb.detach().cpu().numpy()
    S_emb_np = np.ascontiguousarray(S_emb_np.astype(np.float32))

    # distances_s 和 faiss_indices_s 的形状都是 [B, topk]
    distances_s, faiss_indices_s = searcher.search(S_emb_np, k=topk)
    
    # [关键] 使用 id_map 将 Faiss 索引 [B, k] 转换为 原始 article_id [B, k]
    original_article_ids_s = id_map[faiss_indices_s]
    
    # 将 user_id (Tensor) 转换为 NumPy
    user_id_np = user_id.detach().cpu().numpy() # [B,]

    # --- 2. 创建 S_token 的 DataFrame (宽格式) ---
    # user_id_np.shape -> (B,)
    # original_article_ids_s.shape -> (B, k)
    
    # 将 user_id 扩展为 [B, 1] 以便拼接
    user_id_col = user_id_np.reshape(-1, 1)
    
    # 水平拼接 user_id 和 article_ids
    # data_s.shape -> [B, 1 + k]
    data_s = np.hstack([user_id_col, original_article_ids_s])
    
    # 定义列名
    # [ 'user_id', 'item_1', 'item_2', ... ]
    s_cols = ['user_id'] + [f'article_{i+1}' for i in range(topk)]
    
    temp_s_df = pd.DataFrame(data_s, columns=s_cols)
    
    # 拼接 S_dataframe
    S_dataframe = pd.concat([S_dataframe, temp_s_df], ignore_index=True)

    
    # --- 3. 处理 NS_token 的召回结果 (逻辑完全相同) ---
    NS_emb_np = NS_emb.detach().cpu().numpy()
    NS_emb_np = np.ascontiguousarray(NS_emb_np.astype(np.float32))
    
    distances_ns, faiss_indices_ns = searcher.search(NS_emb_np, k=topk)
    
    original_article_ids_ns = id_map[faiss_indices_ns]
    
    # 水平拼接 user_id 和 article_ids
    # user_id_col 还是用上面那个 [B, 1] 的
    data_ns = np.hstack([user_id_col, original_article_ids_ns])
    
    # 定义列名
    ns_cols = ['user_id'] + [f'article_{i+1}' for i in range(topk)]
    
    temp_ns_df = pd.DataFrame(data_ns, columns=ns_cols)

    # 拼接 NS_dataframe
    NS_dataframe = pd.concat([NS_dataframe, temp_ns_df], ignore_index=True)
    
    # 返回更新后的 DataFrames
    return S_dataframe, NS_dataframe


import math
import torch
from torch.optim.optimizer import Optimizer
def get_cosine_schedule_with_warmup_custom(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    start_lr_factor: float = 0.1, # 起始学习率相对于峰值学习率的比例
    end_lr_factor: float = 0.8,   # 结束学习率相对于峰值学习率的比例：1正为峰值；0为衰减到0
    last_epoch: int = -1,
):
    
    def lr_lambda(current_step: int):
        # Warmup 阶段
        if current_step < num_warmup_steps:
            progress = float(current_step) / float(max(1, num_warmup_steps))
            # 线性插值，从 start_lr_factor 上升到 1.0
            return start_lr_factor + (1.0 - start_lr_factor) * progress

        # Cosine Annealing (退火) 阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # 计算标准的余弦衰减因子 (从 1.0 到 0.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # 将 [0, 1] 的余弦衰减范围映射到 [end_lr_factor, 1.0] 的新范围
        return end_lr_factor + (1.0 - end_lr_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

from torch.nn import functional as F
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.08, reduction='mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, seq_embs_norm, pos_embs_norm, neg_embs_norm, loss_mask):
        """
        Args:
            seq_embs (Tensor): 用户序列的表征, 形状 (B, L, H)
            pos_embs (Tensor): 正样本的表征, 形状 (B, L, H)
            neg_embs (Tensor): 负样本的表征, 形状 (B, L, 1, H)
            loss_mask (Tensor): 需要计算损失的位置掩码, 形状 (B, L)
            action_weights (Tensor, optional): 每个有效样本的权重, 形状 (N_valid,)
        """

        hidden_size = seq_embs_norm.size(-1)

        # L2 归一化（移到外部）
        
        # 1. 计算正样本相似度 (B, L) -> (B, L, 1)
        pos_logits = (seq_embs_norm * pos_embs_norm).sum(dim=-1, keepdim=True)

        # 2. 构建全局负样本池并计算负样本相似度
        # neg_embs (B, L, 1, H) -> neg_embedding_all (B*L, H)
        neg_embedding_all = neg_embs_norm.reshape(-1, hidden_size)
        
        # 计算每个用户表征与全局负样本池的相似度
        # (B, L, H) matmul (H, B*L) -> (B, L, B*L)
        neg_logits = torch.matmul(seq_embs_norm, neg_embedding_all.transpose(-1, -2))

        # 3. 拼接正负logits
        # (B, L, 1) cat (B, L, B*L) -> (B, L, 1 + B*L)
        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        # 4. 对padding item 做mask，不计算loss
        # loss_mask (B, L), logits (B, L, 1+B*L) -> (N_valid, 1+B*L)
        valid_logits = logits[loss_mask.bool()]

        #  应用温度系数
        valid_logits = valid_logits / self.temperature

        # 6. 正样本位置为0(正确的logit,错误1的logit,错误2的logit，...)，构建label
        labels = torch.zeros(valid_logits.size(0), device=valid_logits.device, dtype=torch.int64)
        
        # 7. 计算交叉熵损失
        unweighted_loss = F.cross_entropy(valid_logits, labels, reduction='none')
        final_loss = None

        #如果没有权重，则使用原始的聚合方式
        if self.reduction == 'mean':
            final_loss = unweighted_loss.mean()
        elif self.reduction == 'sum':
            final_loss = unweighted_loss.sum()
        else:
            final_loss = unweighted_loss
        return {
            'loss': final_loss,
            'pos_logits_mean': pos_logits[loss_mask.bool()].mean().detach(),
            'neg_logits_mean': neg_logits[loss_mask.bool()].mean().detach(),
        }


import pandas as pd
import numpy as np

def MRR(recall_df, ground_truth_dict, topk=5):
    """
    计算召回数据框的 MRR 
    
    假定 recall_df 是 "宽格式":
        user_id | item_1 | item_2 | ... | item_k
        
    Args:
        recall_df (pd.DataFrame): 包含 user_id 和 top-k 召回物品的 DataFrame。
        ground_truth_dict (dict): {user_id: true_item_id} 的字典。
        topk (int): 要评估的召回深度 (K值)。
        
    Returns:
        pd.DataFrame: 增加了 'MRR' 的原始 DataFrame。
    """
                         
    # --- 1. 数据准备 ---
    
    # 1.1 将 {user_id: true_item_id} 映射到 DataFrame 中，方便对齐
    # 对于召回了但不在 ground_truth 中的 user (例如测试集)，填充 -1
    recall_df['true_item'] = recall_df['user_id'].map(ground_truth_dict).fillna(-1).astype(np.int64)

    # 1.2 获取召回结果的 NumPy 数组 (B, k)
    recall_cols = [f'article_{i+1}' for i in range(topk)]
    recalled_data = recall_df[recall_cols].values
    
    # 1.3 获取真实答案的 NumPy 数组，并 reshape 以便广播 (B, 1)
    true_data = recall_df['true_item'].values.reshape(-1, 1)

    # --- 2. 计算命中矩阵 ---
    
    # 2.1 使用 NumPy 广播比较，得到 (B, k) 的布尔矩阵
    # hits_matrix[i, j] = True 如果第 i 个用户的第 j 个召回 命中了答案
    hits_matrix = (recalled_data == true_data)
    
    # --- 3. 计算标准 MRR ---
    
    # 3.1 找到第一个命中的位置 (rank)
    # np.argmax 在全 False 的行上返回 0，这正是我们想要的
    # +1 是因为 rank 是从 1 开始的
    ranks = np.argmax(hits_matrix, axis=1) + 1 # (B,)
    
    # 3.2 计算 MRR 得分
    mrr_scores = 1.0 / ranks
    
    # 3.3 [关键] 处理完全没命中的情况 (即 hits_matrix 对应行全为 False)
    # np.any(..., axis=1) 找到所有至少命中 1 次的行
    # ~ (取反) 找到所有一次都没命中的行
    no_hits_mask = ~np.any(hits_matrix, axis=1)
    mrr_scores[no_hits_mask] = 0.0
    
    recall_df['MRR'] = mrr_scores
    return recall_df



