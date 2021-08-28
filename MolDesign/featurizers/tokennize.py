from tokenizers import Tokenizer,Regex
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import WhitespaceSplit,Split,Sequence
from tokenizers.models import WordLevel
from tokenizers.processors import BertProcessing
from tokenizers.trainers import WordLevelTrainer
# from .SMILES import CANONICAL_SMILES_PATTERN


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
    def __init__(self,regex=CANONICAL_SMILES_PATTERN,seq_len=10,additional_special_tokens=(),mode=0):
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

if __name__=="__main__":
    tokenizer=Tokenizer.from_file("ChEMBL.json")
    tokenizer.enable_padding(length=11)
    print(tokenizer.encode("c1ccccc1").tokens)
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
