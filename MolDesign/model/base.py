import pytorch_lightning as pl
import torch.nn as nn
import torch


class MLP(nn.Module):
    '''
    This part can be used for any single input and single output layer!
    all operations are not inplace
    '''
    def __init__(self, *seq):
        super(MLP, self).__init__()
        assert len(seq) >= 1, "The MLP layer must have over 1 layer!"
        activate = {"relu": nn.ReLU,
                    "gelu": nn.GELU,
                    "tanh": nn.Tanh,
                    "selu": nn.SELU,
                    "silu": nn.SiLU,
                    "sigmoid": nn.Sigmoid,
                    "ln": nn.LayerNorm,
                    "ln1": nn.BatchNorm1d,
                    "bn2": nn.BatchNorm2d,
                    "bn3": nn.BatchNorm3d,
                    "embed":nn.Embedding}

        def transform(s, *args, **kwargs):
            return activate[s](*args, **kwargs)

        def identify_nn(n):
            '''
            TODO: need to overload this function later?
            '''
            if type(n)==tuple:
                if type(n[0]) == int and type(n[1]) == int:
                    return nn.Linear(*n)
                elif type(n[0]) == str and type(n[1]) == tuple:
                    return transform(n[0], *n[1])
                elif type(n[0]) == str and type(n[1]) == tuple and type(n[2])==dict:
                    return transform(n[0], *n[1], **n[2])
                elif type(n[0]) == str and type(n[1]) == dict:
                    return transform(n[0], **n[1])
                elif type(n[0]) == str and type(n[1]) not in [type, dict]:
                    return transform(*n)
            elif type(n) == float:
                return nn.Dropout(n)
            elif type(n) == str:
                return activate[n]()

        self.mlp = nn.Sequential(*[identify_nn(n) for n in seq])
    def forward(self, input):
        return self.mlp(input)



class Plus(nn.Module):
    def __init__(self, F, plus_idx=0):
        super().__init__()
        self.plus_idx = plus_idx
        self.F = F

    def __repr__(self):
        return "PLus(" + self.F.__repr__() + ")"

    def forward(self, *args, **kwargs):
        return self.F(*args, **kwargs) + args[self.plus_idx]


class Select(nn.Module):
    def __init__(self, F, out_idx=0):
        super().__init__()
        self.F = F
        self.out_idx = out_idx

    def __repr__(self):
        return "Select(" + self.F.__repr__() + f",{self.out_idx})"

    def forward(self, *args, **kwarg):
        return self.F(*args, **kwarg)[self.out_idx]


def GetClones(module, N):
    return nn.Sequential(*[module for i in range(N)])


class FNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,activation:nn.Module=nn.ReLU()):
        super().__init__()
        self.l=nn.Linear(in_features,out_features,bias)
        self.a=activation

    def __repr__(self):
        return self.l.__repr__().replace("Linear(","FNN(").replace(")",", activation="+self.a.__repr__()+")")

    def forward(self,input):
        return self.a(self.l(input))






class PositionEmbedding(nn.Module):
    '''
    shape(S,N,C)  seq_len, batch_size, vocab_size
    '''

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.register_buffer("position", torch.arange(seq_len).unsqueeze(-1))
        self.embed = nn.Embedding(seq_len, embed_dim)

    def __repr__(self):
        return "Position" + self.embed.__repr__()

    def forward(self, input):  # must add input
        return self.embed(self.position)












