# %%
import pandas as pd
import string

# Importing Natural Language Processing toolkit
import nltk

# Downloading the NLTK english stop words
nltk.download("stopwords")

# Importing the NLTK english stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# %% read data
# x = pd.read_csv("reviews.csv")
# Sample DataFrame
x = pd.DataFrame(
    {"review": ["This is a great product! Love aaaaaa", "I didn't like the service."]}
)

# %% lowercase
x["review"] = x["review"].str.lower()

# %% remove punctuation
x["review"] = x["review"].str.translate(str.maketrans("", "", string.punctuation))

# %% tokenization and removeing stopwords
english_stopwords = stopwords.words("english")
english_stopwords

x["review_token"] = x["review"].apply(word_tokenize)
# x["review"]

# %% ###########################


# Ensure nltk punkt tokenizer is available
nltk.download("punkt")

tokens = word_tokenize(x["review"][0])

# Apply tokenization
x["review_token"] = x["review"].apply(word_tokenize)

print(x)

# %% # %% ###########################

# %% remove stopwords
# Importing the NLTK english stop words
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")
english_stopwords


def remove_stopwords(token):
    return [t for t in token if t not in english_stopwords]


x["cleaned_tokens"] = x.review_token.apply(remove_stopwords)
x["cleaned_tokens"]

# %% join back
x["cleaned_tokens_joined"] = x["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))


# %%
