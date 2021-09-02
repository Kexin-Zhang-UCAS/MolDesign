# from MolDesign.model.base import GPT2AR,mydb

# import json
# tokenizer=Tokenizer.from_file("chemts.json")
# a=json.loads(tokenizer.to_str())
# print(a)
# txt2pt("250k_rndm_zinc_drugs_clean.smi","chemts.pt",tokenizer,max_len=82)


import torch
from MolDesign.model.LM.GPT2 import GPT2AR
from MolDesign.model.loader import loader
from MolDesign.featurizers.SMILES.tokenize import SmiTokenizer,txt2pt
from pytorch_lightning import Trainer
def train_tokenizer():
    smit=SmiTokenizer()
    smit.train(["chemts.txt"])
    smit.save("chemts.json",pretty=True)

def generate_db():
    smit=SmiTokenizer.from_file("chemts.json")
    txt2pt("chemts.txt","chemts.pt",max_len=83,tokenizer=smit)

if __name__=="__main__":
    # train_tokenizer()
    # generate_db()

    data=loader("chemts.pt",batch_size=600)
    model=GPT2AR(83,67,128,4,4)
    # print(model.hparams)
    trainer=Trainer(gpus=1,precision=32,auto_lr_find=True,default_root_dir="./check/",max_epochs=8)
    trainer.fit(model,data)

