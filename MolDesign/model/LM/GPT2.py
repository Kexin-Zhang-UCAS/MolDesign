import torch
import torch.nn as nn
from ..base import Plus,Select,PositionEmbedding,GetClones,MLP
import pytorch_lightning as pl



class GPT2Block(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, block_hidden_ratio=4,dropout=0.1):
        super().__init__()
        self.register_buffer("attn_mask", torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1))
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = Plus(Select(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)))
        self.ln2 = Plus(MLP(("ln", embed_dim), (embed_dim, embed_dim * block_hidden_ratio),
                            "gelu", (embed_dim * block_hidden_ratio, embed_dim), dropout))

    def forward(self, input):
        output = self.ln1(input)
        output = self.attn(output, output, output, attn_mask=self.attn_mask)
        output = self.ln2(output)
        return output

class GPT2AR(pl.LightningModule):
    """
    GPT2-AutoRegression Model
    The input database should all have {[BOS]: 0, [EOS]:1, [UNK]:2, [PAD]:3}
    GPT-2 from `language Models are Unsupervised Multitask Learners <https://d4mucfpksywv.cloudfront.net/
    better-language-models/language-models.pdf>`_
    Implementation contributed by: Kexin Zhang
    """

    def __init__(self, seq_len: int,
                 vocab_size: int,
                 embed_dim: int = 128,
                 num_blocks: int = 4,
                 num_heads: int = 4,
                 block_hidden_ratio: int = 4,
                 ignore_index= 3,
                 lr=1e-3,
                 dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Embedding(self.hparams.vocab_size, self.hparams.embed_dim),
            Plus(PositionEmbedding(self.hparams.seq_len - 1, self.hparams.embed_dim)),
            nn.Dropout(self.hparams.dropout),
            GetClones(GPT2Block(self.hparams.embed_dim, self.hparams.num_heads, self.hparams.seq_len - 1,
                                self.hparams.block_hidden_ratio,self.hparams.dropout),self.hparams.num_blocks,),
            nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size)
        )
        self.criteria=nn.CrossEntropyLoss()
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),lr=self.hparams.lr)

    def training_step(self, batch,batch_idx):
        target=batch[:,1:]
        input=batch.transpose(0,1)[:-1]
        output=self.net(input).permute([1,2,0])
        loss=self.criteria(output,target)
        # print(output[0],target[0])
        acc=((output.softmax(dim=1).argmax(dim=1)==target)& (target!=3)).float().mean()/(1-(target==3).float().mean())
        # acc=((output.argmax(dim=1)-target==0) & (target!=3)).float().mean()/(1-(target==1).float().mean())
        self.log("acc",acc,prog_bar=True)
        return {"loss":loss,"acc":acc}

    # def validation_step(self, batch,batch_idx):
    #     target=batch[:,1:]
    #     input=batch.transpose(0,1)[:-1]
    #     output=self.net(input).permute([1,2,0])
    #     loss=self.criteria(output,target)
    #     acc=((output.argmax(dim=1)-target==0) & (target!=1)).float().mean()/(1-(target==1).float().mean())
    #     self.log("acc",acc,prog_bar=True)
    #     return {"loss":loss,"acc":acc}
    #
    # def test_step(self, batch,batch_idx):
    #     target=batch[:,1:]
    #     input=batch.transpose(0,1)[:-1]
    #     output=self.net(input).permute([1,2,0])
    #     loss=self.criteria(output,target)
    #     acc=((output.argmax(dim=1)-target==0) & (target!=1)).float().mean()/(1-(target==1).float().mean())
    #     self.log("acc",acc,prog_bar=True)
    #     return {"loss":loss,"acc":acc}

    # def forward(self,data,possi=False):
    #     assert data.dim==2,"The input seq verctor should have 2 dimensions (batch_size,seq_len)"
    #     input = data.transpose(0, 1)[:-1]
    #     output = self.net(input).permute([1, 2, 0])
    #     if possi:
    #         return output
    #     output=output.argmax(dim=1)
    #     return output