# import torch
# import torch.nn as nn
# from collections import OrderedDict
#
# '''
# For a seq dataset[B,L,H]
# B: Batch size
# L: Longest Sequence Length
# H: Hidden Feature size
# '''
#
# class RNN_Seq2Seq(nn.Module):
#     def __init__(self, input_size:int, hidden_size:int, num_layers:int=2, bias:bool=True, batch_first:bool=True,
#                  dropout:float=0., bidirectional:bool=False, type:str="LSTM"):
#         super(RNN_Seq2Seq, self).__init__()
#         RNN_type_dict = {
#             "RNN": nn.RNN,
#             "LSTM": nn.LSTM,
#             "GRU": nn.GRU,
#             "rnn": nn.RNN,
#             "lstm": nn.LSTM,
#             "gru": nn.GRU,}
#         # TODO: For original RNN, the nonlinearity is another parm which can be set to "relu".
#         #       Judge whether to update this parameter later
#         self.RNN = RNN_type_dict[type](
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bias=bias,
#             batch_first=batch_first,
#             dropout=dropout,
#             bidirectional=bidirectional)
#
#     def forward(self, input):
#         output = self.RNN(input)[0]
#         return output
#
# def Dense(in_features: int, out_features: int, bias: bool = True,dropout:float=0.,inplace=False,activation=nn.ReLU()):
#     return nn.Sequential(OrderedDict([
#             ("Dropout",nn.Dropout(dropout,inplace=inplace)),
#             ("Linear",nn.Linear(in_features=in_features, out_features=out_features, bias= bias)),
#             ("Activation",activation)
#         ]))
#
# '''
# Args:
#         d_model: the number of expected features in the encoder/decoder inputs (default=512).
#         nhead: the number of heads in the multiheadattention models (default=8).
#         num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
#         num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
# '''
#
# class Transformer_Autoencoder(nn.Module):
#     def __init__(self,input_size,dim_feedforward=2048,nhead=1,num_encoder_layers=6,num_decoder_layers=6,dropout=0.1,activation="relu"):
#         super().__init__()
#         self.Transformer=nn.Transformer(
#             d_model=input_size,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward
#         )
#     def forward(self,input): # input (B,)
#         return self.Transformer()
#
#
#
# def Seq2Seq():
#     net=nn.Sequential(OrderedDict([
#         ("RNN",RNN_Seq2Seq(100,100)),
#         ("Dense1",Dense(100,10,dropout=0.1)),
#         ("Dense2",Dense(10,1))
#     ]))
#     return net
#
# # a=Seq2Seq()
# # print(a)
# import torch.nn.functional as F
# # print(F)
# # a=torch.tensor([[0,1,2,3],
# #                 [0,4,6,2]])
# a=torch.LongTensor([[0,1,4,6,5,3,2,9],
#                     [0,2,1,4,3,1,7,2],
#                     [0,1,2,3,4,5,6,7]])
# i=F.one_hot(a,10).transpose(0,1).float()
# print(i[1:].shape)
# mod=nn.Transformer(d_model=10,nhead=2)
# opt=torch.optim.Adam(params=mod.parameters())
# for id in range(10000):
#     opt.zero_grad()
#     out=mod(src=i[:-1],tgt=i[:-1])
#     loss=nn.CrossEntropyLoss()(out.transpose(0,1).transpose(1,2),a[:,1:])
#     loss.backward()
#     opt.step()
#     print(loss)
# torch.save(mod,"1")
# mod=torch.load("1")
# mod.eval()
# a=mod(i[:,0:1],i[:1,0:1]).softmax(dim=-1)
# print(a.topk(4,dim=2))
#
# # print(a[:,1:].shape)
# # print(mod(src=i[:-1],tgt=i[:-1]).transpose(0,1).transpose(1,2).shape)
#
# # print(torch.cuda.is_available())
# # # print(i.shape)
