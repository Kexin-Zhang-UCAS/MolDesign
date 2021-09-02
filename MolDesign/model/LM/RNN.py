import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..base import Select

class GRU(pl.LightningModule):
    def __init__(self,  vocab_size: int,
            embed_dim: int = 128,):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Embedding(self.hparams.vocab_size, self.hparams.embed_dim),
            Select(nn.GRU(self.hparams.embed_dim,self.hparams.embed_dim*2,2,dropout=0.1)),
            nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size)
        )
        self.criteria = nn.CrossEntropyLoss(ignore_index=1)
    def training_step(self, batch,batch_idx):
        target = batch[:, 1:]
        input = batch.transpose(0, 1)[:-1]
        output = self.net(input).permute([1, 2, 0])

        loss = self.criteria(output, target)
        acc1 = ((output.argmax(dim=1) - target == 0) & (target != 1)).float().mean()
        acc2 = (target == 1).float().mean()
        acc = acc1 / (1 - acc2)
        self.log("acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}