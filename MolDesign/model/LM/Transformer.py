import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math


class GPT2_self_attention_sublayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input):
        output = self.norm(input + self.dropout(self.attention(input, input, input)[0]))
        return output


class GPT2_FFN_sublayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=512, dropout=0., activation=nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input):
        output = self.norm(input + self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(input))))))
        return output


class GPT2_decoder_layer(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=512, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        self.GPT2_self_attention = GPT2_self_attention_sublayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.GPT2_FNN = GPT2_FFN_sublayer(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                                          activation=activation)

    def forward(self, input):
        return self.GPT2_FNN(self.GPT2_self_attention(input))


def module_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GPT2_decoder(nn.Module):
    def __init__(self, d_model, layer_num=12, nhead=1, dim_feedforward=512, dropout=0.1, activation=nn.ReLU(),
                 norm=None,batch_first=True):
        super().__init__()
        self.batch_first=batch_first
        self.use_norm=norm
        self.norm = nn.LayerNorm(d_model)
        self.layers = module_clones(GPT2_decoder_layer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation=activation), N=layer_num)

    def forward(self,input):
        output=input
        if self.batch_first:
            output=output.transpose(0,1)
        for mod in self.layers:
            output=mod(input)
        if self.use_norm is not None:
            output = self.norm(output)
        return output


class GPT2_position_embedding(nn.Module):
    def __init__(self):
        super().__init__()




a=torch.tensor([[0,1,2,3,4,5,6,7,0],
                [0,2,3,1,3,4,8,0,0]])
a1=a[:,:-1]
a2=a[:,1:]
b=nn.Embedding(9,12)(a1)
# c=nn.MultiheadAttention(embed_dim=12, num_heads=1)(b,b,b)
# print(c[0].shape,c[1].shape)
# GPT2=GPT2_decoder(12)
# print(GPT2(b))
# print(GPT2_FFN_sublayer(12,1)(b).shape)
# print(nn.MultiheadAttention(12,1)(b,b,b))
# print(GPT2_self_attention_sublayer(d_model=12,nhead=1)(b))
print(GPT2_decoder(12)(b).shape)



