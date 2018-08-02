from string import punctuation

import spacy
from spacy import tokenizer


def normalise_file():
    FILENAME = "shakespeare.txt"

    with open(FILENAME) as f:
        data = f.read()

    data = data.lower()

    for p in punctuation:
        data = data.replace(p, "")

    data = data.replace("\n", " ")
    for _ in range(10):
        data = data.replace("  ", " ")

    with open(FILENAME, "w") as f:
        f.write(data)


# normalise_file()


nlp = spacy.load("en_vectors_web_lg")

with open("datasets/dracula.txt", encoding="utf8") as f:
    data = f.read()

doc = nlp(data)
print(len(doc.vocab))
tok = tokenizer.Tokenizer(nlp.vocab)

# print(nlp.vocab["superstition"].vector.shape)
