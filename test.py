# from MolDesign.model.datasets import Seq
# from MolDesign.model import custom
# import torch.nn as nn
#
# a=nn.Sequential(
#     custom.RNN(10, 10),
#     nn.Linear(10,1),
#     nn.ReLU()
# )
from MolDesign.featurizers.SMILES.ops import to_canonical

print(to_canonical("fwegu"))