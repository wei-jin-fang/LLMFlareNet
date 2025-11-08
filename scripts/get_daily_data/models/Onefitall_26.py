from math import sqrt
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from peft import get_peft_model, LoraConfig, TaskType
# def set_seed(seed=42):
#     """设置所有随机种子确保实验可复现"""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#

# set_seed(3014)
#
# def random_zero_percent(model: torch.nn.Module,percent):
#     """
#     将 model 内所有参数（权重+偏置，任意形状）随机置零。
#     原地操作，不返回新模型。
#     """
#     with torch.no_grad():
#         for param in model.parameters():
#             # 生成同形状均匀随机掩码
#             mask = torch.rand_like(param) < percent
#             param[mask] = 0.0

def random_reinit_percent(model: torch.nn.Module, percent,mode: str = "gaussian"):
    """
    将 model 内所有参数随机重新初始化。
    mode = "gaussian"  -> 标准正态
    mode = "same"      -> 与原始张量同 mean/std
    """
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.rand_like(param) < percent
            if mode == "gaussian":
                param[mask] = torch.randn_like(param)[mask]
            elif mode == "same":                       # 保持原分布
                mean, std = param.mean(), param.std()
                param[mask] = torch.normal(mean, std, size=param.shape, device=param.device, dtype=param.dtype)[mask]
            else:
                raise ValueError("mode must be 'gaussian' or 'same'")
class Onefitall_26Model(nn.Module):
    def __init__(self, args):
        super(Onefitall_26Model, self).__init__()

        self.bert_config = BertConfig.from_pretrained('/media/admin123/Elements/tiaocan/pre_train_model/bert')
        self.bert_config.num_hidden_layers = args.bert_num_hidden_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        self.llm_model = BertModel.from_pretrained(
            '/media/admin123/Elements/tiaocan/pre_train_model/bert',
            trust_remote_code=True,
            local_files_only=True,
            config=self.bert_config,
        )
        # 只解冻LayerNorm参与训练
        for name, param in self.llm_model.named_parameters():
            if "LayerNorm" in name:
                param.requires_grad = True

            else:
                param.requires_grad = False

        # with torch.no_grad():
        #     zero_mask_emb = (torch.rand_like(self.llm_model.embeddings.word_embeddings.weight) < 0.2)  # 比例可以小一些
        #     #置0嵌入层，遗忘学到的词汇
        #     self.llm_model.embeddings.word_embeddings.weight[zero_mask_emb] = 0
        # print(self.llm_model)
        # bili = 0.2
        # for layer in self.llm_model.encoder.layer:
        #     with torch.no_grad():
        #         # 破坏FFN
        #         zero_mask_intermediate = (torch.rand_like(layer.intermediate.dense.weight) < bili)
        #         layer.intermediate.dense.weight[zero_mask_intermediate] = 0
        #         zero_mask_output = (torch.rand_like(layer.output.dense.weight) < bili)
        #         layer.output.dense.weight[zero_mask_output] = 0
        #
        #         # 选择置零比例，例如20%，使注意力失焦
        #         zero_mask = (torch.rand_like(layer.attention.self.query.weight) < bili)
        #         # 置0QKV
        #         layer.attention.self.query.weight[zero_mask] = 0
        #         layer.attention.self.key.weight[zero_mask] = 0
        #         layer.attention.self.value.weight[zero_mask] = 0
        #         # 置0投影矩阵
        #         zero_mask_out = (torch.rand_like(layer.attention.output.dense.weight) < bili)
        #         layer.attention.output.dense.weight[zero_mask_out] = 0
        #
        #         # # 破坏FFN
        #         # zero_mask_intermediate = (torch.rand_like(layer.intermediate.dense.weight) < 0.2)
        #         # layer.intermediate.dense.weight[zero_mask_intermediate] = 0
        #         # zero_mask_output = (torch.rand_like(layer.output.dense.weight) < 0.2)
        #         # layer.output.dense.weight[zero_mask_output] = 0
        # with torch.no_grad():
        #     # 置0嵌入层，遗忘学到的词汇
        #     zero_mask_emb = (torch.rand_like(self.llm_model.embeddings.word_embeddings.weight) < bili)  # 比例可以小一些
        #     self.llm_model.embeddings.word_embeddings.weight[zero_mask_emb] = 0

        # random_zero_percent(self.llm_model,0.2)
        random_reinit_percent(self.llm_model,1.0,'same')
        self.d_model = args.d_model
        self.patch_embedding = PatchEmbedding(args.d_model, patch_len=1, stride=1, dropout=args.dropout)
        self.classification_head = ClassificationHead(args)
        # 添加 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        print("初始化结束")

    def forward(self, inputs):
        # inputs: [batch, 40, 10]
        # Patch 嵌入
        input_patchs, patch_num = self.patch_embedding(inputs)  # [batch, 40, d_model]

        # 输入 BERT 模型
        nlp = self.llm_model(inputs_embeds=input_patchs).last_hidden_state  # [batch, 40, 768]
        # 分类头
        x = self.classification_head(nlp)
        # 添加 Sigmoid 激活函数
        x = self.sigmoid(x)  # 形状: [batch_size, 1]
        return x


class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()
        # 延迟初始化 batch_norm，nlp_last_dim 将在 forward 中动态获取
        self.batch_norm = None
        self.batch_norm64 = nn.BatchNorm1d(args.batch_norm64_dim)
        self.batch_norm32 = nn.BatchNorm1d(args.batch_norm32_dim)
        self.final_dropout = nn.Dropout(args.dropout_rate)
        self.flatten = nn.Flatten()
        # 延迟初始化 fc64，input_dim 将在 forward 中动态计算
        self.fc64 = None
        self.fc32 = nn.Linear(args.fc64_dim, args.fc32_dim)
        self.outlinear = nn.Linear(args.fc32_dim, args.output_dim)
        # 存储 args 中的维度参数
        self.fc64_dim = args.fc64_dim

        self.direct1 = None
        self.output_dim=args.output_dim
    def forward(self, x):
        # x 的形状: [batch_size, patch_num, nlp_last_dim]，例如 [16, 40, 768]

        # 动态获取 nlp_last_dim 和 patch_num
        batch_size, patch_num, nlp_last_dim = x.shape

        if self.direct1 is None:
            input_dim = nlp_last_dim * patch_num
            self.direct1 = nn.Linear(input_dim, self.output_dim).to(x.device)

        x = self.flatten(x)  # 形状: [batch_size, patch_num * nlp_last_dim]
        x = self.final_dropout(x)
        x = self.direct1(x)  # 形状: [batch_size, output_dim]

        return x

class FeatureEmbedding(nn.Module):
    """
    纯 Token 内部建模：学习一个时间步内 (10 个特征) 的交互
    """
    def __init__(self, input_dim=10, d_model=768, hidden_dim=64, dropout=0.1):
        super(FeatureEmbedding, self).__init__()
        # 特征交互 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        # 特征注意力：为每个特征分配重要性权重
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)  # 归一化到特征维度
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]  e.g. [B, 40, 10]

        # Step 1: 特征注意力
        attn = self.attention(x)  # [B, 40, 10]
        x = x * attn  # 加权特征

        # Step 2: 特征交互 MLP
        x = self.mlp(x)  # [B, 40, d_model]

        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Conv1d(
            in_channels=10,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = self._init_pos_embedding(max_len=40, d_model=d_model)
        self.to(torch.float32)

    def _init_pos_embedding(self, max_len, d_model):
        # 正弦-余弦位置编码
        position = torch.arange(max_len).unsqueeze(1)  # [40, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2]
        pe = torch.zeros(max_len, d_model)  # [40, 768]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, 40, 768]
        return pe  # 注册为 buffer 后可自动移到正确设备

    def forward(self, x):
        # x: [batch, 40, 10]
        B, T, N = x.shape  # T=40, N=10
        assert T % self.patch_len == 0, f"Input length {T} must be divisible by patch_len {self.patch_len}"
        x = x.permute(0, 2, 1)  # [batch, 10, 40]
        x = self.value_embedding(x)  # [batch, d_model, 40]
        x = x.permute(0, 2, 1)  # [batch, 40, d_model]

        # 添加位置编码
        pos_embedding = self.pos_embedding.to(x.device)  # [1, 40, d_model]
        x = x + pos_embedding  # [batch, 40, d_model]


        return self.dropout(x), x.shape[1]  # 返回 [batch, 40, 768], patch_num=40

