import transformers
from tokenizers.implementations import CharBPETokenizer


tokenizer=CharBPETokenizer()
tokenizer.train(files=["1/1.txt"],special_tokens=[])
print(tokenizer.get_vocab())
print(tokenizer.encode("c1ccccc1").tokens)

# import io
# config=transformers.GPT2Config(
# )
# print(config)


# a=transformers.GPT2Tokenizer.from_pretrained("./1")
# print(a)