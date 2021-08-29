from tokenizers import Tokenizer,Regex
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import WhitespaceSplit,Split,Sequence
from tokenizers.models import WordLevel
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer
# from .SMILES import CANONICAL_SMILES_PATTERN
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer

ELEMENTS=[
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Al","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Td","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm"
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Uuu","Uub"
]
CANONICAL_SMILES_PATTERN="\[.+?\]|"+"|".join(ele for ele in ELEMENTS if len(ele)==2)+"|."



class SMILES_tokenizer_config():
    def __init__(self,regex=CANONICAL_SMILES_PATTERN,seq_len=1,additional_special_tokens=(),mode=0):
        '''
        TODO: add different post process mode [whether to add BOS EOS CLS SEP MASK etc.]
        :param regex:
        :param seq_len:
        :param additional_special_tokens:
        :param mode:
        '''
        self.regex=regex
        self.seq_len=seq_len
        self.additional_special_tokens=list(additional_special_tokens)
        self.tokenizer=Tokenizer(WordLevel(vocab={},unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), Split(Regex(self.regex), behavior="isolated")])
        self.tokenizer.post_processor = BertProcessing(("[EOS]", 1), ("[BOS]", 0))
        self.trainer = WordLevelTrainer(special_tokens=["[BOS]", "[EOS]", "[UNK]", "[PAD]"] + self.additional_special_tokens)
        self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=self.seq_len)

    def train(self,files):
        self.tokenizer.train(files,trainer=self.trainer)

    def save(self,file):
        self.tokenizer.save(file)

from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from transformers import GPT2LMHeadModel,AdamW,GPT2Config
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm

def txt2pt(txt,pt,tokenizer,max_len=256):
    tokenizer.enable_padding(length=max_len+1,pad_id=3)
    tokenizer.enable_truncation(max_len+1)
    with open(txt,"r") as f:
        torch.save(torch.tensor([i.ids for i in tokenizer.encode_batch(f.read().split("\n"))],dtype=torch.int16),pt)



class gpt2_dataset(Dataset):
    def __init__(self,pt):
        super().__init__()
        self.pt=torch.load(pt)

    def __len__(self):
        return len(self.pt)

    def __getitem__(self, idx):
        return self.pt[idx]


tokenizer=Tokenizer.from_file("ChEMBL.json")
    # # tokenizer.encode_batch()
    # batch_sentences=["c1ccccc1","CCCCCCCCCCCCCCC","C(Br)CCC"]
config=GPT2Config(vocab_size=tokenizer.get_vocab_size(),
                    n_positions=256,
                    n_ctx=256,
                    n_embd=128,
                    n_layer=4,#12
                    n_head=4,# 8
                    n_inner=128*4,
                    activation_function="gelu",
                    resid_pdrop=0.1,
                    embd_pdrop=0.1,
                    attn_pdrop=0.1,
                    layer_norm_epsilon=1e-5,
                    initializer_range=0.02,)
device = torch.device("cuda")

def train():
    model=GPT2LMHeadModel(config).to(device)
    # print(model)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fct = CrossEntropyLoss(ignore_index=3)  # ignores padding token for loss calculation
    train_loader=DataLoader(gpt2_dataset("ChEMBL.pt"),batch_size=400,shuffle=True)
    epoch_id = 0
    # loss_history = []
    try:
        parms = torch.load("gpt2.parms.pt")
        model.load_state_dict(parms["model"])
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        optimizer.load_state_dict(parms["optimizer"])
        epoch_id = parms["epoch_id"]
        # loss_history = parms["loss_history"]
    except:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()
    for epoch in range(10):
        for batch_id,data in enumerate(train_loader):
            data=data.long().to(device)
            input,target=data[:,:-1],data[:,1:]
            output=model(input)
            loss=loss_fct(output[0].transpose(1,2),data[:,1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch:{:3d} Batch:{:3d} loss: {:.4f}".format(epoch,batch_id,loss.item()))
        epoch_id+=1
    model_parameters={
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch_id":epoch_id,
        # "loss_history":loss_history,
    }
    torch.save(model_parameters,"gpt2.parms.pt")
if __name__=="__main__":
    # print("{:.2f} {:f}".format(10,10))
    train()
    pass


    # batch=tokenizer(batch_sentences)
    # txt2pt("SMILES/ChEMBL.txt","ChEMBL.pt",tokenizer)
    # a=torch.load("ChEMBL.pt")

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









import deepsmiles

# deepsmiles.converter
#
# canonical_
#
# def atomicwise_split(smi):
#
#
