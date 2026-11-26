"""Step 1: download paper titles + abstracts from arXiv -> papers.csv"""
import arxiv
import pandas as pd

CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.CV"]
PER_CATEGORY = 1000  # 4 x 1000 = ~4000 papers

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
rows = []

for cat in CATEGORIES:
    print(f"Fetching {cat} ...")
    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=PER_CATEGORY,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    for r in client.results(search):
        rows.append({
            "id": r.entry_id,
            "title": " ".join(r.title.split()),
            "abstract": " ".join(r.summary.split()),
            "authors": ", ".join(a.name for a in r.authors[:3]),
            "year": r.published.year,
            "category": r.primary_category,
            "url": r.entry_id,
        })

df = pd.DataFrame(rows).drop_duplicates("id")
df.to_csv("papers.csv", index=False)
print(f"Saved {len(df)} papers to papers.csv")