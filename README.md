# Semantic Paper Search (Sentence-BERT)

```
arXiv  -->  fetch_papers.py  -->  papers.csv
                                     |
                              build_index.py  (SBERT)
                                     |
                              embeddings.npy
                                     |
query --> SBERT --> cosine similarity --> top-k papers  (app.py)
```

## Run locally
```
pip install -r requirements.txt
python fetch_papers.py      # ~3 min
python build_index.py       # ~2-5 min on CPU
streamlit run app.py
```

## Deploy (free)
1. Push the whole folder to GitHub (include papers.csv and embeddings.npy, ~10 MB).
2. Go to share.streamlit.io -> New app -> pick repo -> main file: app.py
