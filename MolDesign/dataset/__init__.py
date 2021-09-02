# from rdkit import Chem
#
# f2=open("ChEMBL.txt","w")
# f=open("ChEMBL.csv","r")
# for i in f:
#     a=Chem.MolFromSmiles(i.split(";")[-1].strip("\"'\n"))
#     if a is not None:
#         b=Chem.MolToSmiles(a)
#         print(b,file=f2)
#         print(b)
