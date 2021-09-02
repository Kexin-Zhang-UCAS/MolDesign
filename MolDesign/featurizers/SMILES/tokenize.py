from tokenizers import Tokenizer,Regex
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import WhitespaceSplit,Split,Sequence
from tokenizers.models import WordLevel
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer
from tokenizers.implementations import BaseTokenizer
from . import CANONICAL_SMILES_PATTERN
import torch

class SmiTokenizer(BaseTokenizer):
    def __init__(self,
                 vocab = None,
                 regex=CANONICAL_SMILES_PATTERN,
                 unk_token = "[UNK]",
                 bos_token = "[BOS]",
                 eos_token = "[EOS]",
                 pad_token = "[PAD]",
                 additional_special_tokens=tuple()):
        # Initialization of SMILES tokenizer
        self.special_tokens=[bos_token, eos_token, unk_token, pad_token] + list(additional_special_tokens)
        model=WordLevel(vocab=vocab,unk_token=unk_token)
        tokenizer=Tokenizer(model)
        tokenizer.normalizer=NFD()
        tokenizer.pre_tokenizer=Sequence([WhitespaceSplit(), Split(Regex(regex), behavior="isolated")])
        tokenizer.post_processor = BertProcessing((eos_token, 1), (bos_token, 0))
        tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
        super().__init__(tokenizer,parameters=None)

    @staticmethod
    def from_file(path):
        return Tokenizer.from_file(path=path)

    def train(self,files,vocab_size=30000,min_frequency=0,show_progress=True):
        """ Train the model using the given files """

        trainer = WordLevelTrainer(vocab_size=vocab_size,
                                   min_frequency=min_frequency,
                                   show_progress=show_progress,
                                   special_tokens=self.special_tokens)
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)



def txt2pt(txt,pt,max_len,tokenizer):
    tokenizer.enable_padding(length=max_len, pad_id=3)
    tokenizer.enable_truncation(max_len)
    with open(txt,"r") as f:
        db_i = torch.tensor([i.ids for i in tokenizer.encode_batch(f.read().split("\n"))], dtype=torch.int8)
        db=db_i[(db_i == 1).any(dim=1) & (db_i != 2).all(dim=1)]
        torch.save(db,pt)


# t=SmiTokenizer()
# t.train("ChEMBL.txt",vocab_size=64)
# t.save("chembl.json",pretty=True)
# tokenizer=SmiTokenizer.from_file("chembl.json")
# txt2pt("ChEMBL.txt","chembl.pt",max_len=128,tokenizer=tokenizer)
#
# data=torch.load("chembl.pt")
# print(data.shape)


#
#
# ##############################################################
#
# from torch.utils.data import Dataset,DataLoader
# import torch
# import numpy as np
# from transformers import GPT2LMHeadModel,AdamW,GPT2Config
# from torch.utils.tensorboard import SummaryWriter
# import torch
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from tqdm import tnrange, tqdm
#
# def txt2pt(txt,pt,tokenizer,max_len=256):
#     tokenizer.enable_padding(length=max_len+1,pad_id=3)
#     tokenizer.enable_truncation(max_len+1)
#     with open(txt,"r") as f:
#         torch.save(torch.tensor([i.ids for i in tokenizer.encode_batch(f.read().split("\n"))],dtype=torch.int16),pt)
#
#
#
# class gpt2_dataset(Dataset):
#     def __init__(self,pt):
#         super().__init__()
#         self.pt=torch.load(pt)
#
#     def __len__(self):
#         return len(self.pt)
#
#     def __getitem__(self, idx):
#         return self.pt[idx]


# tokenizer=Tokenizer.from_file("ChEMBL.json")
    # # tokenizer.encode_batch()
    # batch_sentences=["c1ccccc1","CCCCCCCCCCCCCCC","C(Br)CCC"]
# config=GPT2Config(vocab_size=tokenizer.get_vocab_size(),
#                     n_positions=256,
#                     n_ctx=256,
#                     n_embd=128,
#                     n_layer=4,#12
#                     n_head=4,# 8
#                     n_inner=128*4,
#                     activation_function="gelu",
#                     resid_pdrop=0.1,
#                     embd_pdrop=0.1,
#                     attn_pdrop=0.1,
#                     layer_norm_epsilon=1e-5,
#                     initializer_range=0.02,)
# device = torch.device("cuda")

# def train():
#     model=GPT2LMHeadModel(config).to(device)
#     # print(model)
#     optimizer = AdamW(model.parameters(), lr=1e-3)
#     loss_fct = CrossEntropyLoss(ignore_index=3)  # ignores padding token for loss calculation
#     train_loader=DataLoader(gpt2_dataset("ChEMBL.pt"),batch_size=400,shuffle=True)
#     epoch_id = 0
#     # loss_history = []
#     try:
#         parms = torch.load("gpt2.parms.pt")
#         model.load_state_dict(parms["model"])
#         model = torch.nn.DataParallel(model, device_ids=[0, 1])
#         optimizer.load_state_dict(parms["optimizer"])
#         epoch_id = parms["epoch_id"]
#         # loss_history = parms["loss_history"]
#     except:
#         model = torch.nn.DataParallel(model, device_ids=[0, 1])
#     model.train()
#     for epoch in range(1):
#         for batch_id,data in enumerate(train_loader):
#             data=data.long().to(device)
#             input,target=data[:,:-1],data[:,1:]
#             output=model(input)
#             loss=loss_fct(output[0].transpose(1,2),data[:,1:])
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print("Epoch:{:3d} Batch:{:3d} loss: {:.4f}".format(epoch,batch_id,loss.item()))
#         epoch_id+=1
#     model_parameters={
#         'model': model.module.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         "epoch_id":epoch_id,
#         # "loss_history":loss_history,
#     }
#     torch.save(model_parameters,"gpt2.parms.pt")
# if __name__=="__main__":
    # # print("{:.2f} {:f}".format(10,10))
    # train()
    # pass
    # tokenizer=Tokenizer.from_file("../ChEMBL.json")
    # print(tokenizer.to_str())
    # batch=tokenizer(batch_sentences)

    # tokenizer.enable_padding(length=9,pad_id=3)
    # tokenizer.enable_truncation(9)
    # print(torch.tensor([i.ids for i in tokenizer.encode_batch(batch_sentences)],dtype=torch.int16))

    # tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    # print(tokenizer.get_vocab())
    # tokenizer.enable_padding(length=11)
    # print(tokenizer.encode(3))
# st=SMILES_tokenizer()
# st.train(["SMILES/ChEMBL.txt"])
# st.save("ChEMBL.json")
#
# print(a.tokenizer.to_str())









