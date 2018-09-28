import logging

logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP


def load_data(file_path):
    return pd.read_csv(file_path)


def get_NPs(text):
    def text_is_good(text):
        if "Voir avis" in text:
            return False
        elif len(text) < 20:
            return False
        else:
            return True

    from nltk.tree import Tree

    if not text_is_good(text):
        logging.info("Subject is not good enough. NPs cannot be extracted.")
        return []

    noun_phrases = []
    # nlp = StanfordCoreNLP(r'/data/stanford/stanford-corenlp-full-2018-02-27', lang='fr')
    nlp = StanfordCoreNLP('http://localhost', port=9000, lang="fr")
    tree = nlp.parse(text)
    ntree = Tree.fromstring(tree)
    for st in ntree.subtrees():
        if st.label() == "NP":
            noun_phrases.append(" ".join(st.leaves()))

    # nlp.close()
    return noun_phrases


def add_splitted_lines(df: pd.DataFrame):
    df["Objet_splitted"] = df["Objet"].dropna().apply(lambda x: x.split("\n"))
    return df

def get_requested_docs(subject_text):
    subject_NPs = get_NPs(subject_text)

    pass


if __name__ == '__main__':
    df = load_data("../notebook/cada-2018-09-25.csv")
    df = add_splitted_lines(df)
    get_requested_docs(df.Objet.iloc[10])
