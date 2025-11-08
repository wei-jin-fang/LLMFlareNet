from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


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
        # self.project_to_768 = nn.Linear(d_model, 768)  # 映射到 768 维度
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

        # x = self.project_to_768(x)  # [batch, 40, 768]
        x =self.dropout(x)
        return x, x.shape[1]  # 返回 [batch, 40, 768], patch_num=40


class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()
        self.flatten = nn.Flatten()
        self.direct1 = None
        self.final_dropout=nn.Dropout(args.dropout_rate)
        self.output_dim=args.output_dim

    def forward(self, x):
        # x 的形状: [batch_size, patch_num, nlp_last_dim]，例如 [16, 40, 768]
        # 动态获取 nlp_last_dim 和 patch_num
        batch_size, patch_num, nlp_last_dim = x.shape
        if self.direct1 is None:
            input_dim = nlp_last_dim * patch_num
            self.direct1 = nn.Linear(input_dim, self.output_dim).to(x.device)
        x = self.flatten(x)  # 形状: [batch_size, patch_num * nlp_last_dim]

        x = self.direct1(x)
        x = self.final_dropout(x)
        # 形状: [batch_size, output_dim]

        return x


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()


        self.d_model = args.d_model
        self.patch_embedding = PatchEmbedding(args.d_model, patch_len=1, stride=1, dropout=args.dropout)


        self.depth=args.bert_num_hidden_layers

        self.multihead_attn = nn.MultiheadAttention(args.d_model, args.n_heads, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(args.d_model)
        self.fc1 = nn.Linear(args.d_model, args.fc64_dim)  # 使用 Trans_fc1 作为 FFN 的扩展维度
        self.fc2 = nn.Linear(args.fc64_dim, args.d_model)  # 回到 d_model

        self.dropout1 = nn.Dropout(args.dropout_rate)  # 使用 Trans_dropout1
        self.dropout2 = nn.Dropout(args.dropout_rate)  # 使用 Trans_dropout2

        self.classification_head = ClassificationHead(args)
        # 添加 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        print("初始化结束")

    def forward(self, model_inoput):
        # inputs: [batch, 40, 10]
        # Patch 嵌入
        inputs, patch_num = self.patch_embedding(model_inoput)  # [batch, 40, d_model]
        x=inputs
        for _ in range(self.depth):
            # 多头注意力
            x1, _ = self.multihead_attn(x, x, x)  # [batch, 40, d_model]

            # 残差连接
            x2 = x1 + x  # [batch, 40, d_model]

            # BatchNorm（对特征维度归一化）
            x3 = x2.transpose(1, 2)  # [batch, d_model, 40]
            x3 = self.batch_norm(x3)  # [batch, d_model, 40]
            x3 = x3.transpose(1, 2)  # [batch, 40, d_model]

            # 前馈网络
            x4 = self.fc1(x3)  # [batch, 40, Trans_fc1]
            x4 = F.relu(x4)
            x4 = self.dropout1(x4)
            x4 = self.fc2(x4)  # [batch, 40, d_model]
            x4 = self.dropout2(x4)

            # 残差连接
            x5 = x4 + x2  # [batch, 40, d_model]
            x = x5

            # 分类头
        x = self.classification_head(x)  # [batch, output_dim]
        x = self.sigmoid(x)  # [batch, output_dim]
        return x
