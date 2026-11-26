"""Semantic Paper Search: SBERT vs keyword search, side by side."""
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Semantic Paper Search", page_icon="🔎", layout="wide")

EXAMPLES = [
    "how do models learn from very few examples",
    "making language models smaller and faster",
    "detecting fake or misleading content",
    "teaching machines using human feedback",
]


@st.cache_resource
def load():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("papers.csv")
    embeddings = np.load("embeddings.npy")
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)
    return model, df, embeddings, tfidf, tfidf_matrix


model, df, embeddings, tfidf, tfidf_matrix = load()
has_category = "category" in df.columns

# ---------- Sidebar ----------
with st.sidebar:
    st.header("How it works")
    st.graphviz_chart(
        """
        digraph {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
            Q [label="Your question"];
            S [label="SBERT\\n(meaning -> numbers)"];
            M [label="Compare with\\nall paper vectors"];
            R [label="Top matching papers"];
            Q -> S -> M -> R;
        }
        """
    )

    st.header("Filters")
    top_k = st.slider("Number of results", 3, 15, 5)

    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    if y_min < y_max:
        year_range = st.slider("Year", y_min, y_max, (y_min, y_max))
    else:
        year_range = (y_min, y_max)

    if has_category:
        cats = st.multiselect("Category (empty = all)", sorted(df["category"].unique()))
    else:
        cats = []

    compare = st.checkbox("Compare with keyword search", value=True)

mask = df["year"].between(*year_range).to_numpy()
if cats:
    mask &= df["category"].isin(cats).to_numpy()


# ---------- Helpers ----------
def top_results(scores, k, min_score):
    s = np.where(mask, scores, -np.inf)
    idx = np.argsort(-s)[:k]
    return [i for i in idx if s[i] > min_score]


def show(idx, scores):
    for i in idx:
        row = df.iloc[i]
        st.markdown(f"**[{row['title']}]({row['url']})**")
        extra = f" · {row['category']}" if has_category else ""
        st.caption(f"{row['authors']} · {row['year']}{extra}")
        st.progress(float(min(max(scores[i], 0.0), 1.0)), text=f"match {scores[i]:.0%}")
        with st.expander("Abstract"):
            st.write(row["abstract"])


def set_query(q):
    st.session_state["query"] = q


# ---------- Main page ----------
st.title("🔎 Semantic Paper Search")
st.caption(f"Search {len(df):,} arXiv papers by **meaning**, not just keywords.")

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_query, args=(ex,), use_container_width=True)

query = st.text_input("Or ask in your own words", key="query")

if not query:
    st.info("Pick an example above or type a question.")
else:
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    sem_scores = embeddings @ q_vec
    sem_idx = top_results(sem_scores, top_k, min_score=-1.0)

    if compare:
        kw_scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().ravel()
        kw_idx = top_results(kw_scores, top_k, min_score=0.0)

        left, right = st.columns(2)
        with left:
            st.subheader("🧠 Semantic (SBERT)")
            st.caption("Understands meaning")
            show(sem_idx, sem_scores)
        with right:
            st.subheader("🔤 Keyword (TF-IDF)")
            st.caption("Matches exact words only")
            if kw_idx:
                show(kw_idx, kw_scores)
            else:
                st.warning("No papers share these exact words.")
    else:
        show(sem_idx, sem_scores)