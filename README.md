# GW-Recommendation-Systems-G4
---

## Dataset

[Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)  
Download and place the CSV files in the working directory before running the notebooks. The evaluation notebook will attempt to download via `kagglehub` if files are not found locally.

---

## Setup

```bash
pip install pandas numpy scipy scikit-learn kagglehub
```

Run notebooks in order (01 → 05). Each notebook is self-contained but 05_evaluation.ipynb aggregates results from all models.

---

## Results Summary

| Approach | Precision@10 | Recall@10 | NDCG | Coverage | Diversity | Serendipity |
|---|---|---|---|---|---|---|
| Non-Personalized | 0.0743 | 0.0733 | 0.1025 | 0.0004 | 0.3556 | 0.0000 |
| Collaborative Filtering | 0.0378 | 0.0557 | 0.0594 | 0.2520 | 0.9700 | 0.0054 |
| Content-Based (TF-IDF) | 0.0025 | 0.0020 | 0.0018 | 0.0160 | 0.4183 | 0.8455 |
| Context-Aware | 0.0741 | 0.0739 | 0.0980 | 0.0004 | 0.1200 | 0.1000 |

---

## How to Reproduce

1. Clone the repo
2. Download the Instacart dataset from Kaggle and place CSVs in the root folder
3. Run notebooks 01 through 05 in order
4. Final comparison table is generated in `05_evaluation.ipynb`
