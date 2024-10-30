# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""Local RAG Demo

Demonstrate local RAG simply and quickly,
by using tiny models.

This builds an index of articles then retrieves
relevant chunks ordered by similarity to the question

Usage:

>>> rag('what is the state of Japans economy?')
Japan's economy grew 2.6% overall last year ...
"""

import languagemodels as lm
import kagglehub
import csv

path = kagglehub.dataset_download("jacopoferretti/bbc-articles-dataset")
articles = []

with open(path+'/bbc_text_cls.csv', 'r') as file:
  reader = csv.reader(file)
  next(reader)
  for row in reader:
    if len(articles) == 50:
        break
    articles.append(row[0])

for a in articles:
    # Chunks and generates embeddings
    lm.store_doc(a)

def rag(question: str) -> str:
    ctx = lm.get_doc_context(question)
    return lm.do(ctx + "\n\n" + question)


