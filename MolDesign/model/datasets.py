# import torch
# from typing import List,Tuple,Dict,Union
# from torch.utils.data import Dataset,DataLoader
# import torch.nn.functional as F
# import torch.nn as nn
# test=[[0,1,2,3,4],[0,2,6],[0,5,7,4]]
#
# class Seq(Dataset):
#     def __init__(self,seq_data:List[List[int]],seq_len:Union[None,int]=None):
#
#         super(Seq, self).__init__()
#         # TODO: By default, 0 is the padding value for most task. May need to pad other value in later research.
#         # TODO: The numbers of SMIS tokens I think is less than 255, so I use uint8 here.
#         self.length = torch.tensor([len(i) for i in seq_data])
#         if seq_len!=None:
#             # TODO: Add Assert to compare longest lengh of seq and seq_len
#             seq_data[0]=seq_data[0]+[0]*(seq_len-len(seq_data[0]))
#         self.data=nn.utils.rnn.pad_sequence([torch.tensor(i,dtype=torch.uint8) for i in seq_data],batch_first=True,padding_value=0)
#
#         # TODO: I thought the packing part
#
#         # self.data2=nn.utils.rnn.pack_padded_sequence(self.data,lengths=[5,3,4],batch_first=True,enforce_sorted=False)
#         # self.data3=nn.utils.rnn.pad_packed_sequence(self.data2,batch_first=True)
#
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, item:int):
#         return self.data[item],self.length[item]


# a=Seq(test,seq_len=2)
# # b=a.data
# loader=DataLoader(a,batch_size=2)
# for i in loader:
#     print(i)
# # print(nn.utils.rnn.pack_padded_sequence(b,lengths=[1,2],batch_first=True,enforce_sorted=False))
#




