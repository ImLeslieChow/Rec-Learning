from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

#from dataset import save_emb
import pandas as pd
import math
import os


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask.unsqueeze(1), dropout_p=0#dropout_p=self.dropout_rate if self.training else 0.0
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            #attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None

class FlashMultiHeadAttention_NSToken(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(dim)) 

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.gamma 

class ASIFwithNSTokens(torch.nn.Module):
    """
    ASIFwithNSTokens:结合ASIF与OneTrans的结构。
    
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        user_senet: 用户特征SENet层
        item_senet: 物品特征SENet层
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, args):  #
        super(ASIFwithNSTokens, self).__init__()    
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen  

        #加载id embedding
        self.load_id_embedding()    

        #创建要使用的词表
        #item侧：wordscate(1~4)、articlecate(0~131、132 共133个);冷度和时效性使用一个DNN来embedding化
        self.wordscate_embedding = torch.nn.Embedding(4+1,args.embedding_units,padding_idx=0)
        self.articlecate_embedding = torch.nn.Embedding(133+1,args.embedding_units,padding_idx=133)

        #user侧：click_env(1~4)、click_dev(1~5)、click_os(1~6)、click_region(1~10)、click_type(1~7)
        self.user_sparse_feat = {"user_click_env":4,'user_click_dev':5,'user_click_os':6,"user_click_region":10,"user_click_refer":7}
        self.user_sparse_emb = torch.nn.ModuleDict()
        for key,value in self.user_sparse_feat.items():
            self.user_sparse_emb[key] = torch.nn.Embedding(value+1,args.embedding_units,padding_idx=0)
        
        #item的连续特征embedding化：冷度seq_popularity和时效性seq_created_ts
        self.item_feat_fnn = torch.nn.Sequential(torch.nn.Linear(2, args.hidden_units))  #直接到hidden_unit

        #user的NS_token：user:'user_activity' 和 4个稀疏特征聚合：env、dev、os、region; 仅"user_click_refer"用于参与QKV计算
        self.user_aggregation_query = torch.nn.ModuleDict()
        user_sparse_feat = ["user_click_env",'user_click_dev','user_click_os',"user_click_region"]
        for key in user_sparse_feat:
            self.user_aggregation_querys[key]=torch.nn.Parameter(torch.randn(args.embedding_units))
        self.user_feat_fnn = torch.nn.Sequential(torch.nn.Linear(1+args.embedding_units*4, args.hidden_units))  #NS token直接一步到hidden unit

        #S_token(id、cate)映射到hidden units上
        self.item_id_encode = torch.nn.Linear(self.id_emb_dim,args.hidden_units)
        self.item_wordscate_encode = torch.nn.Linear(args.embedding_units,args.hidden_units)
        self.item_articlecate_encode = torch.nn.Linear(args.embedding_units,args.hidden_units)

        #创建Attetion模块
        #V ~ item_id; Q 和 K：sideinfo~item:id、wordcate、articlecate、user_click_ts、'user_click_refer' | Position
        #args.num_blocks控制层数； 每层都会融合sideinfo后才进行Q和K的计算
        #encoder
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = RMSNorm(args.hidden_units, eps=1e-6) #torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate,
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = RMSNorm(args.hidden_units, eps=1e-6) #torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def load_id_embedding(self):
        #加载已有词表,该词表不需要训练。
        # 获取当前文件（data_loader.py）所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        id_embedding_path = os.path.join(current_dir, "..", "data", "articles_emb.csv")
        emb_df = pd.read_csv(id_embedding_path)
        
        # 2. 获取维度信息
        # Embedding 向量的维度 (总列数 - 1列ID)
        self.id_emb_dim = emb_df.shape[1] - 1
        # 词表大小：最大ID + 1 (因为 ID从0开始)
        num_embeddings = emb_df.iloc[:, 0].max() + 1 #+1 #第一个加1因为从0开始；第二个加1是因为for padding：but不用padding，因为没有没见过的item
        
        # 3. 初始化一个全 0 的 Numpy 矩阵
        # 这样 ID=0 的位置自动就是全 0 向量，适合做 padding_idx
        emb_matrix = np.zeros((num_embeddings, self.id_emb_dim), dtype=np.float32)
        
        # 4. 将 CSV 中的向量填充到矩阵的对应位置
        # 使用 article_id 作为行索引，确保 ID 和向量一一对应
        ids = emb_df.iloc[:, 0].values        # 取出所有 article_id
        vectors = emb_df.iloc[:,1:].values   # 取出所有对应的向量(列从第二列开始，第一列为ids)
        emb_matrix[ids] = vectors             # 高效的 Numpy 索引赋值

        # 5. 创建并冻结 Embedding 层
        # freeze=True 是关键，它会将 requires_grad 设置为 False
        self.item_emb = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix), 
            freeze=True, 
            #padding_idx=0
        )        
    
    