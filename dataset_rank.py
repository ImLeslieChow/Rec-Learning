import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集；测试集的时候再重新调用一遍__init__(args,"test")
    """
    def __init__(self, args,mode="train"):
        """
        初始化数据集
        args包含数据路径等
        """
        self.args = args
        self.data_path = Path(args.data_path)
        self.max_len = args.max_len
        self.mode = mode

        #读取数据: 
        self.user_sequences = pd.read_csv(self.data_path/"train_click_log.csv") if mode=="train" else pd.read_csv(self.data_path/"testA_click_log.csv")
        self.user_sequences = self.user_sequences.sort_values(["user_id","click_timestamp"])
        self.user_sequences = self.user_sequences.rename(columns={'click_article_id':'article_id'})

        if self.mode=="train":
            self.items_id_embed = pd.read_csv(self.data_path/"articles_emb.csv")
            self.item_attr = pd.read_csv(self.data_path/"articles.csv")
        
        #用于随机负采样：item id是连续的
        assert len(self.item_attr) == len(self.items_id_embed)
        self.items_num = len(self.item_attr)

        #执行item侧特征工程与uer侧特征工程
        self.mode = mode
        #self.share_norm = pd.DataFrame() if self.mode=="train" else pd.read_csv(self.data_path/"share_norm.csv")
        if self.mode=="train":
            self.share_norm = pd.DataFrame()
            self.item_feature()
        
        self.user_feature()
        #if self.mode=="train":
        #    #导出csv
        #    self.share_norm.to_csv("share_norm.csv",index=False, encoding='utf-8-sig')
        #    return
        #将数据按每个用户和每个item分好
        self.init_all()

        

    
    def random_neg(self,exist_items):
        neg_id = np.random.randint(0,self.items_num)
        while neg_id in exist_items :
            neg_id = np.random.randint(0,self.items_num)
        return neg_id

    #item侧的特征工程：只需要训练的时候运行,测试的时候不需要重复运行
    def item_feature(self):
        '''
        创建self.item_feat
        '''

        #（1）item流行度
        data = self.user_sequences[['user_id', 'article_id', 'click_timestamp']].copy()
        data.sort_values(['article_id', 'click_timestamp'], inplace=True)
        article_hot = pd.DataFrame(data.groupby('article_id', as_index=False)[['user_id', 'click_timestamp']].\
                               agg({'user_id':"size", 'click_timestamp': {list}}).values, columns=['article_id', 'user_num', 'click_timestamp'])
        #article_hot~item_id列、user_num(item点击总数)和click_timestamp列(每个item的所有点击时间戳组成的列表)

        # 计算被点击时间间隔的均值
        def time_diff_mean(l):
            if len(l) < 2:
                return None
            else:
                #l = sorted(l)
                return np.mean([j-i for i, j in list(zip(l[:-1], l[1:]))])
        article_hot['time_diff_mean'] = article_hot['click_timestamp'].apply(lambda x: time_diff_mean(x))

        # 点击次数取倒数
        article_hot['user_num'] = 1 / (article_hot['user_num']+1e-9)

        #测试时使用训练用的归一化值
        if self.mode=="train":
            self.share_norm['user_num']=[article_hot['user_num'].min(),article_hot['user_num'].max()]
            self.share_norm['time_diff_mean'] =[article_hot['time_diff_mean'].min(),article_hot['time_diff_mean'].max()]
        # 两者归一化
        article_hot['user_num'] = (article_hot['user_num'] - self.share_norm['user_num'][0]) / (self.share_norm['user_num'][1] - self.share_norm['user_num'][0]+1e-9)
        article_hot['time_diff_mean'] = (article_hot['time_diff_mean'] - self.share_norm['time_diff_mean'][0]) / (self.share_norm['time_diff_mean'][1] - self.share_norm['time_diff_mean'][0]+1e-9)
        article_hot['hot_level'] = article_hot['user_num'] + article_hot['time_diff_mean']    
        article_hot['article_id'] = article_hot['article_id'].astype('int')      
        del article_hot['click_timestamp']  
        del article_hot['user_num']
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

        #（2）文章时效性
        article_timeline = self.item_attr[["article_id","created_at_ts"]].copy()
        article_timeline["created_ts"] = (article_timeline["created_at_ts"] - article_timeline["created_at_ts"].min())/(article_timeline["created_at_ts"].max()-article_timeline["created_at_ts"].min()+1e-9)
        del article_timeline['created_at_ts']
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

        # (3) 文章字数归类：数据分析决定~ 短:<159 中：160~218 长：218~600 超长：>600
        article_wordscate =self.item_attr[["article_id","words_count"]].copy()
        bins = [-1, 159, 218, 600, float('inf')]
        labels = [1, 2, 3, 4] # 对应 短, 中, 长, 超长
        article_wordscate['words_cate'] = pd.cut(article_wordscate['words_count'], bins=bins, labels=labels).astype(int) # 转换为整数类型 
        del article_wordscate['words_count']   
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

        #文章字数归类：总共461类，做频数统计发现，按频率排第140位往后点击次数小于100次~小于100次的归为一类 共用一个embedding   
        article_cate = self.user_sequences[['article_id']].copy()
        article_cate = article_cate.merge(self.item_attr, how='left', on=['article_id'])
        if self.mode =="train":
            category_counts = article_cate['category_id'].value_counts()
            top_categories = category_counts[category_counts >= 100].index.tolist()
            self.category_map = {}
            next_new_id = 0
            # 3.1 先给高频类别分配连续 ID 
            for original_cat_id in top_categories:
                self.category_map[original_cat_id] = next_new_id
                next_new_id += 1
            self.unk_cat_id = next_new_id
            #print("文章新的类别类书：",len(self.category_map))
            #print(self.category_map)

        article_cate_mapped=self.item_attr[['article_id','category_id']].copy()
        article_cate_mapped['category_id_mapped'] = article_cate_mapped['category_id'].map(self.category_map).fillna(self.unk_cat_id).astype(int)
        del article_cate_mapped['category_id']
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        
        #拼接结果
        base_df = self.item_attr[['article_id']].copy()
        # 将所有特征表放入列表
        dfs_to_merge = [base_df, article_hot, article_timeline, article_wordscate, article_cate_mapped]
        
        # 使用 reduce 和 left merge 将它们拼合在一起
        from functools import reduce
        self.item_feat = reduce(lambda left, right: pd.merge(left, right, on='article_id', how='left'), dfs_to_merge)

        #填补缺失值：计算item热度可能有未点击过的item~热度最大值(越冷)
        max_hot_level = self.item_feat['hot_level'].max()
        self.item_feat['hot_level'] = self.item_feat['hot_level'].fillna(max_hot_level)


    
    #user侧的特征工程:训练和测试各调用一次
    def user_feature(self):
        '''
        依次处理
        click_timestamp	click_environment	click_deviceGroup	click_os	click_country	click_region	click_referrer_type

        返回 self.user_feat
        提取基于交互的用户上下文特征：
        time_diff, click_environment, click_deviceGroup, click_os, click_region, click_referrer_type

        '''      
        # ========================= (1) 前后时间差 =========================
        self.user_sequences['time_diff'] = self.user_sequences.groupby('user_id')['click_timestamp'].diff()
        if self.mode == "train":
            self.share_norm['time_diff'] = [self.user_sequences['time_diff'].min(),self.user_sequences['time_diff'].max()]
            self.share_norm["time_diff_mean_train"] = [self.user_sequences['time_diff'].mean(),self.user_sequences['time_diff'].mean()]

        # 计算全局平均时间差，用于填充每个用户的第一次点击（diff结果为NaN）
        #“第一个用用户自己的时间差平均值填上”
        user_mean_diff = self.user_sequences.groupby('user_id')['time_diff'].transform('mean')
        self.user_sequences['time_diff'] = self.user_sequences['time_diff'].fillna(user_mean_diff)
        # 2. 兜底填充 (防止用户只有一次点击导致的 NaN，此时 user_mean_diff 也是 NaN)
        global_mean_diff = self.share_norm["time_diff_mean_train"][0]
        fill_val = global_mean_diff if not pd.isna(global_mean_diff) else 0
        self.user_sequences['time_diff'] = self.user_sequences['time_diff'].fillna(fill_val)


        # Log+minmax归一化时间差 (推荐，有助于模型收敛)
        self.user_sequences['time_diff_log'] = np.log1p(self.user_sequences['time_diff'])
        if self.mode == "train":
            # 记录 Log 后数据的极值
            self.share_norm['time_diff_log'] = [
            self.user_sequences['time_diff_log'].min(),
            self.user_sequences['time_diff_log'].max() ]

        # 避免除零风险
        log_min = self.share_norm['time_diff_log'][0]
        log_max = self.share_norm['time_diff_log'][1]
        diff_denom = log_max - log_min + 1e-9

        # 使用 Log 后的值计算最终的 norm 特征：对 Log 后的值进行 Min-Max 归一化
        self.user_sequences['time_diff_norm'] = (self.user_sequences['time_diff_log'] - log_min) / diff_denom

        # ========================= (2) 类别特征归并 (Binning) =========================
        # --- user_click_os ---
        # 原始保留列表: 2, 12, 13, 17, 19, 20
        keep_os = [2, 12, 13, 17, 19, 20]
        # 生成映射字典: {2: 1, 12: 2, 13: 3, 17: 4, 19: 5, 20: 6} #7for 未见过的
        os_map = {orig_id: new_id for new_id, orig_id in enumerate(keep_os, start=1)}
        
        # 应用映射：不在字典中的会变成 NaN，然后填充为 0
        self.user_sequences['click_os_mapped'] = self.user_sequences['click_os'].map(os_map).fillna(7).astype(int)

        # --- user_click_region ---
        keep_regions = [5, 8, 9, 13, 16, 20, 21, 24, 25, 26]
        # 生成映射字典: {5: 1, 8: 2, ..., 26: 10} #11 for 未出现过的
        region_map = {orig_id: new_id for new_id, orig_id in enumerate(keep_regions, start=1)}
        
        self.user_sequences['click_region_mapped'] = self.user_sequences['click_region'].map(region_map).fillna(11).astype(int)

        # ========================= (3) 其他直接使用的特征 无需修改=========================
        # user_click_env: 1~4 user_click_dev: 1~5 user_click_type: 1~7 这些特征可以直接使用，后续送入 Embedding 层

        # ========================= (4) 用户活跃度 =========================
        # 注意：当前逻辑计算的值越大，代表用户越“不活跃” (点击少且间隔大)
        data = self.user_sequences[['user_id', 'article_id', 'click_timestamp']].copy()
        # 确保时间戳有序，这对于计算 time_diff 至关重要
        data.sort_values(['user_id', 'click_timestamp'], inplace=True)

        # 使用标准的 pandas agg 方式，更稳定
        user_act = data.groupby('user_id', as_index=False).agg({
            'article_id': 'size',
            'click_timestamp': list
        }).rename(columns={'article_id': 'click_size'})

        # 计算时间间隔的均值
        def time_diff_mean(l):
            # 如果只有1次点击，间隔无法计算。
            # 返回 0 意味着认为其极其活跃（间隔为0），或者你可以返回一个较大的默认值代表不活跃。
            if len(l) < 2:
                return 0
            else:
                # 这里的 list 已经是排好序的了
                return np.mean([j - i for i, j in zip(l[:-1], l[1:])])

        user_act['time_diff_mean'] = user_act['click_timestamp'].apply(time_diff_mean)

        # --- 特征转换 ---
        # 点击次数取倒数：点击越多，值越小
        # 先转 float 确保精度
        user_act['click_size'] = 1.0 / (user_act['click_size']+1e-9).astype(float)

        # --- 归一化 (Min-Max) ---
        # 记录训练集的极值用于归一化
        if self.mode == "train":
            self.share_norm['user_click_size'] = [user_act['click_size'].min(), user_act['click_size'].max()]
            self.share_norm['user_time_diff_mean'] = [user_act['time_diff_mean'].min(), user_act['time_diff_mean'].max()]

        # 使用 share_norm 进行归一化，并添加 1e-9 防止除零错误
        click_size_min, click_size_max = self.share_norm['user_click_size']
        user_act['click_size'] = (user_act['click_size'] - click_size_min) / (click_size_max - click_size_min + 1e-9)

        time_diff_min, time_diff_max = self.share_norm['user_time_diff_mean']
        user_act['time_diff_mean'] = (user_act['time_diff_mean'] - time_diff_min) / (time_diff_max - time_diff_min + 1e-9)

        # 计算最终活跃度 (实际是不活跃度)
        user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']

        # 清理与格式化
        user_act['user_id'] = user_act['user_id'].astype('int')
        # 只保留需要的列，方便后续 merge
        user_act = user_act[['user_id', 'active_level']]
        self.user_act = user_act
  
    def init_all(self):
        # === 需要在 __init__ 中添加的预处理 ===
        # 1. 将 item 特征转为 numpy 数组以便快速按索引查找
        # 假设最大的 article_id 是 max_aid
        max_aid = self.item_attr['article_id'].max()
        # 初始化一个全 0 (或合适的默认值) 的大数组
        self.item_popularity_lookup = np.zeros(max_aid + 1, dtype=np.float32) #+ 1 因为id从0开始计
        self.item_created_ts_lookup = np.zeros(max_aid + 1, dtype=np.float32)
        self.item_wordscate_lookup = np.zeros(max_aid + 1, dtype=np.int64)
        self.item_articlecate_lookup = np.zeros(max_aid + 1, dtype=np.int64)

        # 填充具体值 (前提是 self.item_feat 已经 merge 好了所有特征)
        # 确保 article_id 是整数索引
        self.item_popularity_lookup[self.item_feat['article_id'].values] = self.item_feat['hot_level'].values
        self.item_created_ts_lookup[self.item_feat['article_id'].values] = self.item_feat['created_ts'].values 
        self.item_wordscate_lookup[self.item_feat['article_id'].values] = self.item_feat['words_cate'].values
        self.item_articlecate_lookup[self.item_feat['article_id'].values] = self.item_feat['category_id_mapped'].values

        # 2. 预先按用户分组好数据，避免 getitem 时临时 groupby
        # 将每个用户的完整历史序列存为一个字典，key是uid，value是各个特征序列
        # 这一步可能比较耗内存，如果内存不够，需要换成存储索引的方式
        self.user_data_cache = {}
        grouped = self.user_sequences.groupby('user_id')
        for uid, group in tqdm(grouped, desc="Caching user sequences"):
            # 确保按时间排序
            group = group.sort_values('click_timestamp')
            self.user_data_cache[uid] = {
                'article_id': group['article_id'].values,
                'time_diff_norm': group['time_diff_norm'].values,
                'click_environment': group['click_environment'].values,
                'click_deviceGroup': group['click_deviceGroup'].values,
                'click_os_mapped': group['click_os_mapped'].values,
                'click_region_mapped': group['click_region_mapped'].values,
                'click_referrer_type': group['click_referrer_type'].values,
                'time_diff': group['time_diff'].values,
                # 非序列特征 (取第一条即可，因为对同一用户是常数)
                'active_level': self.user_act[self.user_act['user_id'] == uid]['active_level'].values[0] 
                                if uid in self.user_act['user_id'].values else self.user_act['active_level'].max(),
                # 其他
                "uer_id":np.int64(uid)
            }
        self.all_user_ids = list(self.user_data_cache.keys())    #torch 用的 从0开始连续值做index  


    def _pad_truncate(self, seq, max_len, pad_val=0, dtype=np.int64):
            """
            辅助函数：对序列进行截断或填充 (Pre-padding: [0, 0, 1, 2, 3])
            """
            seq_len = len(seq)
            if seq_len >= max_len:
                # 截断：取最近的 max_len 个交互 (保持不变)
                return np.array(seq[-max_len:], dtype=dtype)
            else:
                # 填充：前面补 pad_val
                pad_len = max_len - seq_len
                padding = np.full(pad_len, pad_val, dtype=dtype)
                # 将 padding 放在 seq 前面
                return np.concatenate([padding, seq]).astype(dtype)

    def __len__(self):
        return len(self.all_user_ids)

    def __getitem__(self, index,candidate=None):
        """
        获取单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns: 返回numpy类型的数据,且用字典存储{[seq_idemb]:numpy,...}
            item侧序列数据
            seq_id: 用户交互历史的序列的item ID_embedding
            seq_popularity:交互历史序列的item 冷度
            seq_created_ts:交互历史序列的item 创建时间~sincos化还是minmax归一化？————minmax归一化合理
            seq_wordscate:交互历史序列的item 的字数归类(类似长、中、短) 
            seq_articlecate:交互历史序列的item 的类别


            user侧序列数据
            user_click_ts:用户点击历史的log+minmax的前后时间差~用于P_timegap 
            #:暂时不用：user_click_sin：时间周期变量(星期一 和星期七连在一起) vs user_click_weekday:在一周内的哪一天。7天归一化后，周一和周日差别很大。 #时间变量序列先经过time_fnn(内部可以构造各种分桶)稠密化吧
            
            ——————暂时想法：这些变化少，建立词表 和 各建立一个查询变量进行聚合来捕捉变化转变为非序列特征(应该比人为识别是否发生过变化的方式要好)
            user_click_env:用户点击历史的环境 1~4
            user_click_dev:用户点击历史的设备 1~5
            user_click_os：用户点击历史的系统 2~20  2 12 13 17 19 20 其余归为同一类
            user_click_country:用户点击历史的城市 1~11: 1 8  10 11 ————基本没用；不用该特征
            user_click_region：用户点击历史的地区 1~28 ： 5 8 9 13 16 20 21 24 25 26 其余归为同一类
            user_click_type:用户点击历史的来源 1~7
            #除了点击来源容易在同一序列变化~这两个用作为序列信息外,其他用查询变量聚合，作为NS_Token的其中一维

            user侧非序列数据
            user_activity：用户活跃度
            #下面的特征可以依赖于序列建模，暂时不用这些特征。
            user_timeline：用户对文章的时效性偏好
            user_wordscate:用户对文章的长度偏好~求均值~查词表
            是否添加用户兴趣广泛度量？————大部分20个以下？————似乎序列建模能将信息压缩到NStoken上。

            pos: 正样本ID embedding（即下一个真实访问的item）
            neg: 负样本ID（随机抽）
            测试或排序阶段的时候也不需要弄正负样本

            训练的时候最后丢掉最后一戳；排序时最后一戳为“真正的预测item”

            token_mask:标明非padding token、S_tokens、NS_tokens
        """              
        # 注意：PyTorch DataLoader 传进来的是 index，不是直接的 uid
        uid = self.all_user_ids[index]
        user_data = self.user_data_cache[uid]
        
        # 获取该用户的所有交互序列
        full_article_ids = user_data['article_id']
        seq_len = len(full_article_ids)
        
        # [!!! BUG 修复 !!!] 
        # 必须在这里复制一份，否则
        # Label=0 的分支会修改 self.user_data_cache[uid] 的原始数据！
        # 导致下一次 epoch 这个用户的数据是错的。
        full_article_ids = np.copy(full_article_ids)

        # 1. 划分历史序列和预测目标
        if candidate is None:
            # 训练模式： 随机决定正负样本
            
            # [!!! BUG 修复 !!!]
            # np.random.randint(0, 1) 永远返回 0。
            # 应该用 np.random.randint(0, 2) (0或1) 
            # 或者 np.random.rand() (0到1的浮点数)
            if np.random.rand() > 0.5:
                Label = 1 
                # Label=1 时，我们使用真实的下一个 item，即 full_article_ids[-1]
                # 不需要做任何操作
            else:
                Label = 0
                # Label=0 时，我们替换掉最后一个 item
                # [!!! BUG 修复 !!!]
                # set(tuple(full_article_ids[-1])) 是错的, 应该是 set(full_article_ids)
                true_item_set = set(full_article_ids)
                full_article_ids[-1] = self.random_neg(true_item_set)

            # 在训练时，序列是完整的，包含正/负样本在最后
            hist_article_ids = full_article_ids
            hist_time_diff_norm = user_data['time_diff_norm']
            hist_time_diff = user_data['time_diff']
            hist_click_env = user_data['click_environment']
            hist_click_dev = user_data['click_deviceGroup']
            hist_click_os = user_data['click_os_mapped']
            hist_click_region = user_data['click_region_mapped']
            hist_click_referrer = user_data['click_referrer_type']

        else:
            # 测试/验证模式：接入候选item到交互历史上
            hist_article_ids = np.append(full_article_ids, candidate)
            Label = 0 # 占位符 (在评估时我们不知道标签)

            #候选item的对应用户特征用上一个 or (min+max)/2 来填补
            hist_time_diff_norm = user_data['time_diff_norm']
            log_norm_mean = (self.share_norm['time_diff_log'][0] + self.share_norm['time_diff_log'][1]) / 2
            hist_time_diff_norm = np.append(hist_time_diff_norm, log_norm_mean)
            
            hist_time_diff = user_data['time_diff']
            raw_mean = (self.share_norm['time_diff'][0] + self.share_norm['time_diff'][1]) / 2
            hist_time_diff = np.append(hist_time_diff, raw_mean) 
            
            hist_click_env = np.append(user_data['click_environment'], user_data['click_environment'][-1])
            hist_click_dev = np.append(user_data['click_deviceGroup'], user_data['click_deviceGroup'][-1])
            hist_click_os = np.append(user_data['click_os_mapped'], user_data['click_os_mapped'][-1])
            hist_click_region = np.append(user_data['click_region_mapped'], user_data['click_region_mapped'][-1])
            hist_click_referrer = np.append(user_data['click_referrer_type'], user_data['click_referrer_type'][-1])

        # 2. 生成 Item 侧的关联序列特征 (使用 Numpy 快速查找)
        seq_popularity = self.item_popularity_lookup[hist_article_ids]
        seq_created_ts = self.item_created_ts_lookup[hist_article_ids]
        seq_wordscate = self.item_wordscate_lookup[hist_article_ids]
        seq_articlecate = self.item_articlecate_lookup[hist_article_ids]

        # 3. 序列截断与填充
        max_len = self.max_len 
        real_len = min(len(hist_article_ids), max_len)
        
        token_mask = np.zeros(max_len , dtype=np.int32)
        token_mask[-real_len:]=1

        # 统一填充
        data = {
            # --- Item 侧序列 ---
            # [修改] 你的 token_mask 是 L+1，序列应该是 L
            'seq_id': self._pad_truncate(hist_article_ids, max_len, pad_val=0, dtype=np.int64),
            'seq_popularity': self._pad_truncate(seq_popularity, max_len, pad_val=0, dtype=np.float32),
            'seq_created_ts': self._pad_truncate(seq_created_ts, max_len, pad_val=0, dtype=np.float32),
            'seq_wordscate': self._pad_truncate(seq_wordscate, max_len, pad_val=0, dtype=np.int64),
            'seq_articlecate': self._pad_truncate(seq_articlecate, max_len, pad_val=0, dtype=np.int64),

            # --- User 侧序列 ---
            'user_click_ts': self._pad_truncate(hist_time_diff, max_len, pad_val=0, dtype=np.float32),
            "user_click_ts_norm": self._pad_truncate(hist_time_diff_norm, max_len, pad_val=0, dtype=np.float32),
            'user_click_env': self._pad_truncate(hist_click_env, max_len, pad_val=0, dtype=np.int64),
            'user_click_dev': self._pad_truncate(hist_click_dev, max_len, pad_val=0, dtype=np.int64),
            'user_click_os': self._pad_truncate(hist_click_os, max_len, pad_val=0, dtype=np.int64),
            'user_click_region': self._pad_truncate(hist_click_region, max_len, pad_val=0, dtype=np.int64),
            'user_click_refer': self._pad_truncate(hist_click_referrer, max_len, pad_val=0, dtype=np.int64),
            
            # --- 其他信息 ---
            'token_mask': token_mask,
            'user_activity': np.float32(user_data['active_level']), # 标量
            'Label': np.int64(Label),
            'user_id': uid # 有时候也需要返回 uid
        }

        return data

    
    # 排序阶段用的：此时监督指标是一个二分类问题。~预测正确的概率。 越高越放在前面去
    def rank_batch(self, uids, candidate_items):
        """
        为一批 (user, candidate_item) 对生成一个排序批次 (用于评估/测试)。
        
        Args:
            uids (list or np.array): 用户ID列表, [B].
            candidate_items (list or np.array): 对应的候选物品ID列表, [B].

        Returns:
            dict: 一个 collated batch, 准备好送入模型。
        """
        batch_samples = []
        
        # 遍历 B 个 (user, item) 对
        for uid, candidate in zip(uids, candidate_items):
            # 检查用户是否存在
            if uid not in self.user_data_cache:
                print(f"Warning: User {uid} not in cache during ranking. Skipping.")
                continue

            # --- 1. 获取用户数据 ---
            user_data = self.user_data_cache[uid]
            full_article_ids = user_data['article_id']

            # --- 2. 接入候选 item (模拟 "eval" 模式) ---
            hist_article_ids = np.append(full_article_ids, candidate)
            
            # --- 3. 伪造上下文特征 ---
            # (这里的逻辑必须与 __getitem__ 的 'else' 分支完全一致)
            hist_time_diff_norm = user_data['time_diff_norm']
            log_norm_mean = (self.share_norm['time_diff_log'][0] + self.share_norm['time_diff_log'][1]) / 2
            hist_time_diff_norm = np.append(hist_time_diff_norm, log_norm_mean)
            
            hist_time_diff = user_data['time_diff']
            raw_mean = (self.share_norm['time_diff'][0] + self.share_norm['time_diff'][1]) / 2
            hist_time_diff = np.append(hist_time_diff, raw_mean) 
            
            hist_click_env = np.append(user_data['click_environment'], user_data['click_environment'][-1])
            hist_click_dev = np.append(user_data['click_deviceGroup'], user_data['click_deviceGroup'][-1])
            hist_click_os = np.append(user_data['click_os_mapped'], user_data['click_os_mapped'][-1])
            hist_click_region = np.append(user_data['click_region_mapped'], user_data['click_region_mapped'][-1])
            hist_click_referrer = np.append(user_data['click_referrer_type'], user_data['click_referrer_type'][-1])

            # --- 4. 生成 Item 侧特征 ---
            seq_popularity = self.item_popularity_lookup[hist_article_ids]
            seq_created_ts = self.item_created_ts_lookup[hist_article_ids]
            seq_wordscate = self.item_wordscate_lookup[hist_article_ids]
            seq_articlecate = self.item_articlecate_lookup[hist_article_ids]

            # --- 5. 序列截断与填充 ---
            max_len = self.max_len 
            real_len = min(len(hist_article_ids), max_len)
            
            token_mask = np.zeros(max_len , dtype=np.int32)
            token_mask[-real_len:]=1

            # --- 6. 统一填充字典 ---
            data = {
                'seq_id': self._pad_truncate(hist_article_ids, max_len, pad_val=0, dtype=np.int64),
                'seq_popularity': self._pad_truncate(seq_popularity, max_len, pad_val=0, dtype=np.float32),
                'seq_created_ts': self._pad_truncate(seq_created_ts, max_len, pad_val=0, dtype=np.float32),
                'seq_wordscate': self._pad_truncate(seq_wordscate, max_len, pad_val=0, dtype=np.int64),
                'seq_articlecate': self._pad_truncate(seq_articlecate, max_len, pad_val=0, dtype=np.int64),
                
                'user_click_ts': self._pad_truncate(hist_time_diff, max_len, pad_val=0, dtype=np.float32),
                "user_click_ts_norm": self._pad_truncate(hist_time_diff_norm, max_len, pad_val=0, dtype=np.float32),
                'user_click_env': self._pad_truncate(hist_click_env, max_len, pad_val=0, dtype=np.int64),
                'user_click_dev': self._pad_truncate(hist_click_dev, max_len, pad_val=0, dtype=np.int64),
                'user_click_os': self._pad_truncate(hist_click_os, max_len, pad_val=0, dtype=np.int64),
                'user_click_region': self._pad_truncate(hist_click_region, max_len, pad_val=0, dtype=np.int64),
                'user_click_refer': self._pad_truncate(hist_click_referrer, max_len, pad_val=0, dtype=np.int64),
                
                'token_mask': token_mask,
                'user_activity': np.float32(user_data['active_level']),
                'Label': np.int64(0), # 在评估时，Label 设为0占位
                'user_id': uid
            }
            
            batch_samples.append(data)

        # --- 7. 检查是否为空 ---
        if not batch_samples:
            return None # 或者一个空字典

        # --- 8. 调用 collate_fn 打包 ---
        collated_batch = self.collate_fn(batch_samples)
        
        return collated_batch

    #分batch
    @staticmethod
    #def collate_fn(self,batch): 静态方法不需要self参数了
    def collate_fn(batch):
        '''
        重写：主要任务就是把一个 list of dicts 转换成一个 dict of batched tensors。
        自定义 collate_fn: 将多个样本堆叠成一个 batch
        因为 __getitem__ 已经保证了所有样本的结构和形状一致，
        这里只需要简单地将它们 stack 起来并转为 Tensor。
        '''
        batch_data = {}
        # 假设 batch 是一个列表，里面的每个元素都是 __getitem__ 返回的字典 data
        # data 的 keys: 'seq_idemb', 'seq_popularity', ..., 'token_mask', 'pos', 'neg' 等
        
        # 获取字典的所有键 (取第一个样本的键即可)
        keys = batch[0].keys()
        
        for key in keys:
            # 提取当前 key 在所有样本中的值
            # 例如: [sample1['seq_idemb'], sample2['seq_idemb'], ...]
            values = [sample[key] for sample in batch]
            
            # 将 list of numpy arrays 转换为单个 tensor
            # torch.tensor(np.stack(values)) 比直接 torch.tensor(values) 通常更高效且兼容性好
            # 注意：np.stack 会自动处理标量 (stack 后变成 1D tensor) 和数组 (stack 后变成 (B, L) tensor)
            batch_data[key] = torch.from_numpy(np.stack(values))

        return batch_data #[B,L,D]

    