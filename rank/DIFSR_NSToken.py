from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

#from dataset import save_emb
import pandas as pd
import math
import os

# FlashMultiHeadAttention 建立为：FlashMultiHeadAttention_ID(输入X_V和X_QK)、FlashMultiHeadAttention_sideinfo(正常的QKV)[是不是还可以控制每一大层的sideinfo的自注意力机制层数？for双循环]
#其实在forward中传入不同的query、key、value 即可~ forward(x,x,x) qkv均由同一个x生成而来
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

    def forward(self, query, key, value, attn_mask=None,relative_time =None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


        # 降级到标准注意力机制
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) 
        if relative_time is not None:
            scores += relative_time
        scores = scores*scale
            
        if attn_mask is not None: 
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), -torch.inf)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(nan=0.0) #padding位置的权重0/0=NAN问题
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training) #model.eval() 会设置self.training为False
        attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None

#NS_Token用的
class FlashMultiHeadAttention_NSToken(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention_NSToken, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None,relative_time =None):
        batch_size, seq_len_q, _ = query.size()
        _,seq_len_kv,_ = key.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)


        # 降级到标准注意力机制
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) 
        if relative_time is not None:
            scores += relative_time
        scores = scores*scale
            
        if attn_mask is not None: 
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), -torch.inf)

        #attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_weights = attn_weights.nan_to_num(nan=0.0)  #padding位置的权重0/0=NAN问题      
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training) #model.eval() 会设置self.training为False
        attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)
        return output, None

#是否需要换成门控网络
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

class GatedFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, ffn_hidden_dim=None):
        """
        重构为 SwiGLU 门控 FFN (LLaMA-style)。
        使用 Conv1d 来高效实现序列上的线性变换。
        """
        super(GatedFeedForward, self).__init__()

        # LLaMA 为了保持参数量与标准 4x FFN 相似，
        # 使用了 4 * H * (2/3) 作为中间维度。
        if ffn_hidden_dim is None:
            ffn_hidden_dim = int((4 * hidden_units) * (2 / 3))

        # W_up (对应 W1)
        self.conv_up = torch.nn.Conv1d(hidden_units, ffn_hidden_dim, kernel_size=1)
        # W_gate (对应 W3)
        self.conv_gate = torch.nn.Conv1d(hidden_units, ffn_hidden_dim, kernel_size=1)
        # W_down (对应 W2)
        self.conv_down = torch.nn.Conv1d(ffn_hidden_dim, hidden_units, kernel_size=1)
        
        # 激活函数 SiLU (Swish)
        self.activation = torch.nn.SiLU()

        # 沿用你原有的 dropout 结构
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs 形状: (Batch, SeqLen, HiddenDim)
        
        # 转换以适应 Conv1d: (Batch, HiddenDim, SeqLen)
        x = inputs.transpose(-1, -2)

        # 门控路径
        gate = self.conv_gate(x) # (Batch, FFN_HiddenDim, SeqLen)

        # 上采样路径
        hidden = self.conv_up(x) # (Batch, FFN_HiddenDim, SeqLen)
        hidden = self.activation(hidden) # (Batch, FFN_HiddenDim, SeqLen)
        
        # 逐元素相乘 (门控)
        gated = hidden * gate      # (Batch, FFN_HiddenDim, SeqLen)
        
        # 应用第一个 dropout (模仿你的 dropout1)
        gated = self.dropout1(gated)

        # 下采样
        outputs = self.conv_down(gated) # (Batch, HiddenDim, SeqLen)
        
        # 应用第二个 dropout (模仿你的 dropout2)
        outputs = self.dropout2(outputs)

        # 转换回 (Batch, SeqLen, HiddenDim)
        outputs = outputs.transpose(-1, -2)
        
        # [可选] 添加残差连接 (Transformer 块的标准操作)
        # outputs = inputs + outputs
        
        #NAN→0
        outputs = outputs.nan_to_num(nan=0.0)
        
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


#dataset :self.share_norm['time_diff'][1]
class TimeDifferenceEncoder(nn.Module):
    """
    时间差编码器 (修正版)
    功能：
    1. 接收**原始**相邻时间差序列 (未经过 Log 变换)。
    2. 自动恢复绝对时间并计算任意两点 (i, j) 之间的真实绝对时间差。
    3. 对真实时间差进行 Log 分桶。
    4. 通过 Embedding 查找得到时间偏差项 (Time Gap Bias)。
    """
    def __init__(self, num_buckets=128, num_heads=8, max_time_diff=3600*24*30):
        """
        Args:
            num_buckets (int): 分桶数量。
            num_heads (int): Attention 头数。
            max_time_diff (float): 预计的最大时间差（秒），用于缩放 Log 分桶。
                                   超过这个值的将被截断到最后一个桶。
                                   默认约 30 天 (视具体业务调整)。
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        # 计算 log 分桶的缩放系数
        # 我们希望 log(max_time_diff + 1) * scale_factor approx num_buckets - 1
        self.max_log_val = torch.log(torch.tensor(max_time_diff + 1.0))
        #self.max_log_val = torch.tensor(max_time_diff)

        self.time_emb = nn.Embedding(num_buckets, num_heads) #一个向量[x1,x2..] 每一个维度为一个注意力头的时间偏置 
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.01)

    def forward(self, raw_time_diff):
        """
        Args:
            raw_time_diff: (Batch, SeqLen) 原始相邻时间差序列，单位建议为秒。
                           Padding 部分应该是 0。
        Returns:
            time_bias: (Batch, Num_Heads, SeqLen, SeqLen)
        """
        B, S = raw_time_diff.shape
        
        # 1. 恢复绝对时间戳 (Batch, SeqLen)
        # 这一步必须在原始线性空间进行！
        timestamps = torch.cumsum(raw_time_diff, dim=1)

        # 2. 计算任意两点间的真实绝对时间差 (Batch, SeqLen, SeqLen)
        # 利用广播：(B, S, 1) - (B, 1, S)
        raw_dist_matrix = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))

        # 3. Log 分桶 (标准的对数分桶策略)
        # bucket = floor( num_buckets * log(dist + 1) / log(max_dist + 1) )
        # 这样可以保证时间差在 [0, max_time_diff] 区间内均匀映射到 [0, num_buckets]
        log_dist = torch.log1p(raw_dist_matrix)
        
        # 计算分桶 ID：应该是计算到每个时间差在什么范围，然后归类
        # 乘以 (num_buckets - 1) / self.max_log_val 进行缩放
        scale = (self.num_buckets - 1) / self.max_log_val.to(raw_time_diff.device)
        bucket_ids = (log_dist * scale).long()
        
        # 截断超过最大范围的值，防止索引越界
        bucket_ids = torch.clamp(bucket_ids, min=0, max=self.num_buckets - 1)

        # 4. Embedding 查找
        time_bias = self.time_emb(bucket_ids) # (B, S, S, NumHeads)
        time_bias = time_bias.permute(0, 3, 1, 2) # (B, NumHeads, S, S)

        return time_bias
#time_bias = self.time_encoder(batch['user_click_ts'])


class AbsoluteTimeEncoder(nn.Module):
    """
    绝对时间位置编码 (T-APE) - 简化版 v2
    使用正弦/余弦函数对 "自序列开始以来的绝对时间" 进行编码。
    
    输入：
    1. 原始的相邻时间差 (例如单位为秒)。
    2. Padding Mask (True=Padding, False=Real)。
    输出：(B, S, H) 的位置编码，可直接与 Item Embedding 相加
    """
    def __init__(self, hidden_dim, dropout=0.1, max_timescale=10000.0):
        """
        Args:
            hidden_dim (int): 编码的维度 (H)，必须与 Item Embedding 维度相同
            dropout (float): Dropout 比例
            max_timescale (float): 正弦函数中的分母常数，同 Transformer
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        # --- 注册正弦函数的分母 (div_term) ---
        # 1.0 / (max_timescale^(2i / H))
        # (H/2)
        inv_freq = 1.0 / (max_timescale ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        
        # 将其注册为 buffer，它不是参数，但应随模型保存
        # 形状变为 (1, 1, H/2) 以便广播
        self.register_buffer('div_term', inv_freq.unsqueeze(0).unsqueeze(0))

    def forward(self, raw_time_diff, padding_mask):
        """
        Args:
            raw_time_diff: (B, S) 原始的相邻时间差 (单位为秒或分钟等)。
            padding_mask: (B, S)布尔掩码, True 表示该位置是 Padding, False 表示是真实数据。
        Returns:
            abs_time_pe: (B, S, H) 绝对时间位置编码
        """
        B, S = raw_time_diff.shape
        device = raw_time_diff.device

        # --- 1. 恢复绝对时间戳 (B, S) ---
        
        # [关键] 先将 padding 位置的时间差清零，防止它们影响 cumsum
        # (B, S)
        safe_raw_time_diff = raw_time_diff.float().masked_fill(padding_mask, 0.0)
        
        # 直接在安全的时间差上累加
        # (B, S)
        timestamps = torch.cumsum(safe_raw_time_diff, dim=1)
        
        # [关键] 再次使用 mask，将 padding 位置的时间戳强行归零
        # 因为 cumsum 会把最后一个真实时间戳 "传播" 到后续的 padding 位置
        timestamps = timestamps.masked_fill(padding_mask, 0.0)

        # --- 2. 计算正弦编码 ---
        # (B, S, 1) * (1, 1, H/2) -> (B, S, H/2)
        angles = timestamps.unsqueeze(2) * self.div_term

        # 初始化 PE 矩阵 (B, S, H)
        pe = torch.zeros(B, S, self.hidden_dim, device=device)
        
        # 填充偶数维度 (sin)
        pe[..., 0::2] = torch.sin(angles)
        # 填充奇数维度 (cos)
        pe[..., 1::2] = torch.cos(angles)
        
        # --- 3. 最终修正：使用 mask 强制清零 ---
        # padding 位置 (timestamp=0) 对应的 PE 是 [sin(0), cos(0), ...] = [0, 1, 0, 1, ...]
        # 这不是我们想要的 0 向量。
        # 使用 padding_mask (B, S) -> (B, S, 1) 广播清零
        pe = pe.masked_fill(padding_mask.unsqueeze(2), 0.0)
        
        return self.dropout(pe)



class DIFSRwithNSTokens(torch.nn.Module):
    """
    DIFSRwithNSTokens:结合DIF与OneTrans的结构。
    
    Args:
        args: 全局参数

    Attributes:
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        max_time_diff：最大时间差 用于计算两个时间编码

    """

    def __init__(self, args,max_time_diff):  #
        super(DIFSRwithNSTokens, self).__init__()    
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.max_len  
        self.max_time_diff = max_time_diff

        #位置embedding
        #self.pos_emb = torch.nn.Embedding(args.max_len*2 + 1, args.hidden_units, padding_idx=0) # *2避免越界


        #加载id embedding
        self.load_id_embedding()    

        #创建要使用的词表
        #item侧：wordscate(1~4)、articlecate(0~131、132 共133个);冷度和时效性使用一个DNN来embedding化
        self.wordscate_embedding = torch.nn.Embedding(4+1,args.embedding_units,padding_idx=0)
        self.articlecate_embedding = torch.nn.Embedding(133+1,args.embedding_units,padding_idx=133)

        #user侧：click_env(1~4)、click_dev(1~5)、click_os(1~6+1)、click_region(1~10+1)、click_type(1~7)
        self.user_sparse_feat = {"user_click_env":4,'user_click_dev':5,'user_click_os':7,"user_click_region":11,"user_click_refer":7}
        self.user_sparse_emb = torch.nn.ModuleDict()
        for key,value in self.user_sparse_feat.items():
            self.user_sparse_emb[key] = torch.nn.Embedding(value+1,args.embedding_units,padding_idx=0)
        
        #item的连续特征embedding化：冷度seq_popularity和时效性seq_created_ts
        self.item_feat_fnn = torch.nn.Sequential(torch.nn.Linear(2, args.hidden_units),
                                                 RMSNorm(args.hidden_units,eps=1e-6))  #直接到hidden_unit

        #user的NS_token：user:'user_activity' 和 4个稀疏特征聚合：env、dev、os、region; 仅"user_click_refer"用于参与QKV计算
        self.user_aggregation_querys = torch.nn.ParameterDict() #torch.nn.ModuleDict()报错 因为torch.nn.Parameter是一个向量不是模块
        user_sparse_feat = ["user_click_env",'user_click_dev','user_click_os',"user_click_region"]
        for key in user_sparse_feat:
            self.user_aggregation_querys[key]=torch.nn.Parameter(torch.randn(args.embedding_units))
        self.user_feat_fnn = torch.nn.Sequential(torch.nn.Linear(1+args.embedding_units*4, args.hidden_units),RMSNorm(args.hidden_units,eps=1e-6))  #NS token直接一步到hidden unit
        self.user_click_refer_hidden = torch.nn.Sequential(torch.nn.Linear(args.embedding_units, args.hidden_units),RMSNorm(args.hidden_units,eps=1e-6)) 
        
        #S_token(id、cate)映射到hidden units上
        self.item_id_encode = torch.nn.Sequential(torch.nn.Linear(self.id_emb_dim,args.hidden_units),RMSNorm(args.hidden_units,eps=1e-6))
        self.item_wordscate_encode =  torch.nn.Sequential(torch.nn.Linear(args.embedding_units,args.hidden_units),RMSNorm(args.hidden_units,eps=1e-6))
        self.item_articlecate_encode =  torch.nn.Sequential(torch.nn.Linear(args.embedding_units,args.hidden_units),RMSNorm(args.hidden_units,eps=1e-6))

        #创建Attetion模块
        #V ~ item_id; Q 和 K：sideinfo~item:id、wordcate、articlecate、'user_click_refer'  
        #每一层的Q和K的生成均由all sideinfo、V为上一层输出的V
        #args.num_blocks控制层数； 每层都会融合sideinfo后才进行Q和K的计算
        #user_click_ts 用于生成相对位置编码(+到softmax上) 代替了position
        #encoder
        ##for id
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layernorms_qk = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        ##for sideinfo
        self.sideinfo =["seq_id","seq_continuous",'seq_wordscate','seq_wordscate','seq_articlecate','user_click_refer']
        self.attention_layernorms_sideinfo =torch.nn.ModuleList()
        self.attention_layers_sideinfo=torch.nn.ModuleList()
        self.forward_layernorms_sideinfo = torch.nn.ModuleList()
        self.forward_layers_sideinfo = torch.nn.ModuleList()
        self.fusion_fnn = torch.nn.ModuleList()
        #每一层的注意力矩阵的相对时间偏置
        self.relative_time = torch.nn.ModuleList()
        #NS_token
        self.attention_layernorms_ns = torch.nn.ModuleList() 
        self.attention_layers_ns = torch.nn.ModuleList()
        self.forward_layernorms_ns = torch.nn.ModuleList()
        self.forward_layers_ns = torch.nn.ModuleList()
        self.position_ns = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            #id的自注意力机制
            new_attn_layernorm = RMSNorm(args.hidden_units, eps=1e-6) #torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm_qk = RMSNorm(args.hidden_units, eps=1e-6) #torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms_qk.append(new_attn_layernorm_qk)
            

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate,
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = RMSNorm(args.hidden_units, eps=1e-6) #torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = GatedFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            #这里append得appendMoudelDict:{info_key:对应模块}~{wordcate:模块,...}
            temp_attn_layernorms = torch.nn.ModuleDict()
            temp_attn_layers = torch.nn.ModuleDict()
            temp_fnn_layernorms =torch.nn.ModuleDict()
            temp_fnn_layers = torch.nn.ModuleDict()
            for key in self.sideinfo:
                temp_attn_layernorms[key] = RMSNorm(args.hidden_units, eps=1e-6)
                temp_attn_layers[key] =FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
                temp_fnn_layernorms[key] = RMSNorm(args.hidden_units, eps=1e-6)
                temp_fnn_layers[key] = GatedFeedForward(args.hidden_units, args.dropout_rate)
            self.attention_layernorms_sideinfo.append(temp_attn_layernorms)
            self.attention_layers_sideinfo.append(temp_attn_layers)
            self.forward_layernorms_sideinfo.append(temp_fnn_layernorms)
            self.forward_layers_sideinfo.append(temp_fnn_layers)

            #融合层:思考是否需要非线性激活函数？
            self.fusion_fnn.append(torch.nn.Sequential(RMSNorm(args.hidden_units*len(self.sideinfo),eps=1e-6),
                                                       torch.nn.Linear(args.hidden_units*len(self.sideinfo),args.hidden_units),
                                                       torch.nn.GELU(),
                                                       torch.nn.Dropout()))

            #相对时间偏置
            self.relative_time.append(TimeDifferenceEncoder(num_buckets=args.num_buckets, num_heads=args.num_heads, max_time_diff=self.max_time_diff))

            #NS_token的自注意力机制
            new_attn_layernorm = RMSNorm(args.hidden_units, eps=1e-6) 
            self.attention_layernorms_ns.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention_NSToken(
                args.hidden_units, args.num_heads, args.dropout_rate,
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers_ns.append(new_attn_layer)

            new_fwd_layernorm = RMSNorm(args.hidden_units, eps=1e-6) 
            self.forward_layernorms_ns.append(new_fwd_layernorm)

            new_fwd_layer = GatedFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers_ns.append(new_fwd_layer)
        
            self.position_ns.append(AbsoluteTimeEncoder(hidden_dim=args.hidden_units, dropout=args.dropout_rate, max_timescale=self.max_time_diff))
        #二分类头
        self.history_query = torch.nn.Parameter(torch.randn(args.hidden_units))
        self.classifaction_head = torch.nn.Sequential(torch.nn.Linear(args.hidden_units*2,args.hidden_units),
                                                      torch.nn.GELU(),
                                                      RMSNorm(args.hidden_units, eps=1e-6), 
                                                      torch.nn.Dropout(p=args.dropout_rate),
                                                      torch.nn.Linear(args.hidden_units,1))

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

    def feat2emb(self,batch_data):
        '''
        将所有数据变成embedding化(hidden_unit)
        输入：
        data = {
            # --- Item 侧序列 ---
            'seq_id': ,
            'seq_popularity': 
            'seq_created_ts': 
            'seq_wordscate':,
            'seq_articlecate': 

            # --- User 侧序列 ---
            'user_click_ts': 
            "user_click_ts_norm":
            'user_click_env': 
            'user_click_dev': 
            'user_click_os':
            'user_click_region':
            'user_click_refer': 
            
            # --- 其他信息 ---
            'token_mask': token_mask,
            'user_activity':  # 标量
            "pos_seq":
            'neg_seq'
        }
        '''
        batch_data = {k: v.to(self.dev) for k, v in batch_data.items()}
        #（1）item id
        item_mask = (batch_data["token_mask"]==1).to(self.dev)
        #item_mask =item_mask[:, :-1] #token mask 已经含NS_token位置，长度和序列长度不同了。
        #batch_data["seq_id"] = self.item_emb(batch_data["seq_id"]*item_mask)
        batch_data["seq_id"] = self.item_emb(batch_data["seq_id"])
        batch_data["seq_id"] = self.item_id_encode(batch_data["seq_id"])
        #(2)seq_wordscate
        #batch_data["seq_wordscate"] = self.wordscate_embedding(batch_data["seq_wordscate"]*item_mask)
        batch_data["seq_wordscate"] = self.wordscate_embedding(batch_data["seq_wordscate"])
        batch_data["seq_wordscate"] = self.item_wordscate_encode(batch_data["seq_wordscate"])
        #(3)seq_articlecate
        #batch_data["seq_articlecate"] =  self.articlecate_embedding(batch_data["seq_articlecate"]*item_mask)
        batch_data["seq_articlecate"] =  self.articlecate_embedding(batch_data["seq_articlecate"])
        batch_data["seq_articlecate"] = self.item_articlecate_encode(batch_data["seq_articlecate"])
        #(4)item冷度和时效性([B,L] 变为 [B,L,1])
        seq_continuous = torch.concat([batch_data['seq_created_ts'].unsqueeze(2),batch_data['seq_popularity'].unsqueeze(2)],dim=-1)
        batch_data["seq_continuous"] = self.item_feat_fnn(seq_continuous)

        #user侧
        #'user_click_env' 'user_click_dev' 'user_click_os' 'user_click_region' 聚合这4个用于NS_token
        aggregated_features = []
        padding_mask = (batch_data["token_mask"]==0).to(self.dev)
        #padding_mask =  padding_mask[:,:-1]
        for key in ['user_click_env','user_click_dev','user_click_os','user_click_region']:
            seq_emb =  self.user_sparse_emb[key](batch_data[key]) #[B,L,H] 作为Value 和 Key向量 在L维度上聚合
            #batch_data[key] = seq_emb #是否需要储存下来？
            query = self.user_aggregation_querys[key] #nn.Parameter ~一个query向量
            # 计算 Attention Scores
            # (B, L, H) * (H,) -> (B, L)
            # 这里利用广播机制，计算 Query 与序列中每个位置的点积
            scores = torch.matmul(seq_emb, query)
            #处理padding
            #scores = scores.masked_fill(padding_mask, -1e9)  #开了混合精度 ：float16 或半精度浮点数 的数值范围有限~-1e9超出这个范围
            scores = scores.masked_fill(padding_mask, -torch.inf)
            #计算 Attention 权重: (B, L)
            attn_weights = torch.softmax(scores, dim=-1)   
            #加权求和聚合：更高效地用矩阵乘法: (B, 1, L) @ (B, L, H) -> (B, 1, H) 
            aggregated_emb = torch.bmm(attn_weights.unsqueeze(1), seq_emb)  
            #添加到列表中最后concat 
            aggregated_features.append(aggregated_emb)
        aggregated_features.append(batch_data["user_activity"].unsqueeze(1).unsqueeze(2)) #"user_activity" ~[B,] 变为 [B,1,1]
        aggregated_features = torch.concat(aggregated_features,dim=-1)
        batch_data["NS_Token"] = self.user_feat_fnn(aggregated_features) 

        #user_click_refer ~作为Q K 属性之一
        batch_data["user_click_refer"] = self.user_sparse_emb["user_click_refer"](batch_data["user_click_refer"])
        batch_data["user_click_refer"] = self.user_click_refer_hidden(batch_data["user_click_refer"])

        batch_data = {k: v.to(self.dev) for k, v in  batch_data.items()}

        return batch_data
    
    def emb2emb(self,batch_data):
        '''
        接收batch_data的数据进行运算
        '''
        batch_data = self.feat2emb(batch_data)

        #将NS_token拼上序列数据上吧，对齐token_mask 再想想看

        token_mask = batch_data["token_mask"] #[0,0,..1,1,..,1]
        B, L = token_mask.shape
        device = token_mask.device

        # V attn_mask:0和2完全不参与计算 1内部保持因果关系
        # 定义无效位置 (0 和 2)
        #invalid_mask = (token_mask == 0) | (token_mask == 2) # (B, L)
        # 基础因果 mask
        #causal = torch.triu(torch.ones(L, L, device=token_mask.device, dtype=torch.bool), diagonal=1)
        # 合并：因果遮蔽 OR Key无效 OR Query无效
        # (1, L, L) | (B, 1, L) | (B, L, 1) -> (B, L, L) 自动广播
        #attn_mask_id = causal.unsqueeze(0) | invalid_mask.unsqueeze(1) | invalid_mask.unsqueeze(2)

        # True = 可见, False = 不可见
        # 1. 定义有效位置 (只有 1 是有效的)
        valid_mask = (token_mask == 1) # (B, L)
        # 2. 生成基础可见性矩阵 (Query和Key都必须是有效的1)
        # (B, L, 1) & (B, 1, L) -> (B, L, L)
        # 只有 Q是1 且 K是1 的位置才有可能为 True
        base_visible = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        # 3. 加入因果限制 (下三角为True表示可见，上三角为False表示不可见)
        # diagonal=0 包含对角线 (自己看自己)
        causal_visible = torch.tril(torch.ones(L, L, device=token_mask.device, dtype=torch.bool), diagonal=0)
        # 4. 合并: 既要都是1，又要满足因果
        id_attn_mask = base_visible & causal_visible.unsqueeze(0)

        # NS_attn_mask : 2可以看到所有的1 ~ 因为只有一个查询~attn mask 也是一个向量即可
        ns_mask = torch.full((B, 1), 2, device=device, dtype=token_mask.dtype)
        token_mask_2 = torch.cat([token_mask, ns_mask], dim=-1) # 形状: [B, L+1]
        # mask convention: True = 可见
        # 1. 定义 NS Query 能看到的 Key
        #    目标: Key 是 1 或 Key 是 2
        #    等价于: Key 不是 0 (Padding)
        visible_keys = (token_mask_2 != 0) # 形状: [B, L+1]
        # 2. [关键] 调整形状以匹配 (Lq=1, Lk=L+1)
        #    我们需要 [B, 1, L+1]
        NS_attn_mask = visible_keys.unsqueeze(1)

        #padding mask
        padding_mask = (token_mask==0) #[B,L]

        #关键：DIFSR
        ns = batch_data["NS_Token"]
        seq_id = batch_data["seq_id"]
        for i in range(len(self.attention_layernorms)):
            #相对时间偏置
            time_rab = self.relative_time[i](batch_data["user_click_ts"])
            #print(f"time_rab_{i}",time_rab[0])

            #sideinfo 计算输出Q和K
            fuse_emb=[]
            for key in self.sideinfo:
                seqs = batch_data[key]
                #Pre-norm
                if self.norm_first:
                    # layer norm 是自注意力之后才layernorm还是之前？~先layernorm
                    x = self.attention_layernorms_sideinfo[i][key](seqs)
                    mha_outputs, _ = self.attention_layers_sideinfo[i][key](x, x, x, attn_mask=id_attn_mask,relative_time=time_rab)
                    seqs = seqs + mha_outputs #自注意力的残差连接
                    seqs = seqs + self.forward_layers_sideinfo[i][key](self.forward_layernorms_sideinfo[i][key](seqs)) # FNN先norm再残差连接
                
                #Post-norm
                else:
                    mha_outputs, _ = self.attention_layers_sideinfo[i][key](seqs, seqs, seqs, attn_mask=id_attn_mask,relative_time=time_rab)
                    seqs = self.attention_layernorms_sideinfo[i][key](seqs+mha_outputs)
                    seqs = self.forward_layernorms_sideinfo[i][key](seqs + self.forward_layers_sideinfo[i][key](seqs))
                
                fuse_emb.append(seqs)
            fuse_emb = self.fusion_fnn[i](torch.concat(fuse_emb,dim=-1))
            #print(f"fuse_emb_{i}",fuse_emb[0])

            #V~seq_id Q和K:所有信息
            if self.norm_first:
                #Pre-norm
                x_qk = self.attention_layernorms_qk[i](fuse_emb) #规定x_qk肯定先norm；否则post-norm里x_qk没用的上norm
                seq_id = self.attention_layernorms[i](seq_id)
                mha_outputs, _ = self.attention_layers[i](x_qk, x_qk, seq_id, attn_mask=id_attn_mask,relative_time=time_rab)
                seq_id = seq_id + mha_outputs
                seq_id = seq_id + self.forward_layers[i](self.forward_layernorms[i](seq_id))

            #Post-norm
            else:
                x_qk = self.attention_layernorms_qk[i](fuse_emb)
                mha_outputs, _ = self.attention_layers[i](x_qk, x_qk, seq_id, attn_mask=id_attn_mask,relative_time=time_rab)
                seq_id = self.attention_layernorms[i](seq_id+mha_outputs)
                seq_id = self.forward_layernorms[i](self.forward_layers[i](seq_id)+seq_id)
            #print(f"seq_id_{i}",seq_id[0])
            
            #NS_token
            #Pre-norm
            item_embedding_dim =seq_id.shape[-1]
            pos_emb = self.position_ns[i](batch_data["user_click_ts"],padding_mask)
            #print(f"poss_{i}",pos_emb[0])
            seq_id_pos = seq_id*(item_embedding_dim**0.5)+pos_emb
            #print(f"seq_id_pos_{i}",seq_id_pos[0])
            if self.norm_first:
                x_ns_q = self.attention_layernorms_ns[i](ns)
                x_ns_rest = self.attention_layernorms_ns[i](seq_id_pos)
                x_ns_kv = torch.concat([x_ns_rest,x_ns_q],dim=1)#[B,L,H] 在L维度上拼接
                mha_outputs,_ = self.attention_layers_ns[i](x_ns_q,x_ns_kv,x_ns_kv,attn_mask = NS_attn_mask) #是否需要用time rab呢？~ 使用绝对位置编码?
                ns = ns + mha_outputs
                ns = self.forward_layers_ns[i](self.forward_layernorms_ns[i](ns))+ns
            
            else:
                x_ns_kv = torch.concat([seq_id_pos,ns],dim=1) #[B,L,H] 在L维度上拼接
                mha_outputs,_ = self.attention_layers_ns[i](ns,x_ns_kv,x_ns_kv,attn_mask = NS_attn_mask)
                ns = self.attention_layernorms_ns[i](ns +mha_outputs)
                ns =  self.forward_layernorms_ns[i](self.forward_layers_ns[i](ns)+ns)
            #print(f"NS_token_{i}",ns[0])
        

        #先聚合S_token，然后和NS_Token concat一起进入分类头
        scores = torch.matmul(seq_id, self.history_query)
        scores = scores.masked_fill(padding_mask, -torch.inf)
        attn_weights = F.softmax(scores.float(), dim=-1).to(seq_id.dtype)
        attn_weights = attn_weights.nan_to_num(nan=0.0) # 防止 0/0 导致 NaN
        aggregated_s = torch.bmm(attn_weights.unsqueeze(1),seq_id)
        combined_features = torch.cat([aggregated_s,ns], dim=-1)
        combined_features = combined_features.squeeze(1)
        # --- 3. 分类 ---
        # (B, 2*H) -> (B, 1)
        logits = self.classifaction_head(combined_features)

        return logits






            

                    














       

    
    