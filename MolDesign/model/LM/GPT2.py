import torch
import torch.nn as nn
import copy
import math

class GPT2_decoder_self_attention_sublayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input,mask):
        output = self.norm(input + self.dropout(self.attention(input, input, input,attn_mask=mask)[0]))
        return output


class GPT2_decoder_FFN_sublayer(nn.Module):
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
        self.GPT2_self_attention = GPT2_decoder_self_attention_sublayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.GPT2_FNN = GPT2_decoder_FFN_sublayer(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                                          activation=activation)

    def forward(self, input,mask):
        return self.GPT2_FNN(self.GPT2_self_attention(input,mask))


def module_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GPT2_decoder(nn.Module):
    def __init__(self, d_model, layer_num=12, nhead=1, dim_feedforward=512, dropout=0.1, activation=nn.ReLU(),
                 norm=None):
        super().__init__()
        self.use_norm = norm
        self.norm = nn.LayerNorm(d_model)
        self.layers = module_clones(GPT2_decoder_layer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation=activation), N=layer_num)

    def forward(self, input,mask):
        output = input
        for mod in self.layers:
            output = mod(input,mask)
        if self.use_norm is not None:
            output = self.norm(output)
        return output


class positional_encoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class GPT2_embedding(nn.Module):
    '''
    including 1. Embedding
              2. PositionalCoding
    '''
    def __init__(self,num_classes,d_model,max_len,dropout=0.):
        super().__init__()
        self.embed=nn.Embedding(num_classes,d_model)
        self.pos_encode=positional_encoding(d_model=d_model,dropout=dropout,max_len=max_len)

    def forward(self,input):
        return self.pos_encode(self.embed(input))



class GPT2(nn.Module):
    def __init__(self,num_classes,d_model,max_len,nhead=1,dim_feedforward=512,layer_num=12,activation=nn.ReLU(),dropout=0.1):
        super().__init__()
        self.encode=GPT2_embedding(num_classes=num_classes,d_model=d_model,max_len=max_len,dropout=dropout)
        self.decode=GPT2_decoder(d_model=d_model, layer_num=layer_num, nhead=nhead,
                     dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,norm=None)
        self.linear=nn.Linear(d_model,num_classes)

    def generate_square_subsequent_mask(self,sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,input):
        mask=self.generate_square_subsequent_mask(input.size(1))
        ouput=self.encode(input).transpose(0,1)
        # print(ouput.size(0))
        ouput=self.decode(ouput,mask)
        return self.linear(ouput)



# print(generate_square_subsequent_mask(7))


if __name__=='__main__':
    a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0],
                      [0, 2, 3, 1, 3, 4, 8, 0, 0]])
    input = a[:, :-1]
    target = a[:, 1:]
    # b=GPT2_embedding(9,12,20)(input).transpose(0,1)
    # print(GPT2_decoder(12)(b).shape)
    net=GPT2(num_classes=9,d_model=30,max_len=8)
    opt=torch.optim.Adam(params=net.parameters())
    # for i in range(1000):
    #     ouput=net(input)
    #     loss=nn.CrossEntropyLoss()(ouput.permute(1,2,0),target)
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()
    #     print(loss)
    # torch.save(net,"GPT2")
    net=torch.load("GPT2")
    net.eval()
    print(net(torch.tensor([[0,2,3,1,3,4,8]])).argmax(-1))

