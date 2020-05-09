from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'[A-Z,a-z]+')
tokenizer = RegexpTokenizer(r'\w+')

result = tokenizer.tokenize("Wow! I am excited to learn data science")
print(result)