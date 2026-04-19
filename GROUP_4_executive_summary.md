# Executive Summary

**Project:** End-to-End Recommendation System for Online Grocery Shopping  
**Group:** Group 4 — Jacobo Galindo, Javier López-Usero, Samer Roz, Pablo Torres & Paula Candiles  
**Course:** Recommendation Engines · IE University · 2025–26

---

## Overview

This study aims to design, implement, and evaluate four recommendation systems for the Instacart online grocery platform (non-personalized, collaborative filtering, content-based, and context-aware). The dataset used for this project is the Instacart Online Grocery Basket Analysis dataset, with 3.4 million orders, 206,000 users, and 49,000 products.

---

## Dataset

With no missing values, the Instacart dataset offers rich behavioral and product data. The global reorder rate is approximately 59%, users place an average of 16 orders and 10 products in each basket, and 35% of purchases take place on weekends. The interaction space is incredibly sparse (\~99.82%), which poses a major personalization challenge. Using a temporal train/test split that preserved each user's final order as ground truth, approximately 550,000 training and approximately 222,000 test interactions were produced across approximately 21,000 users.

---

## Models & Key Results

| Approach | Precision@10 | Recall@10 | NDCG | Coverage | Diversity | Serendipity |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Non-Personalized (Most Popular) | **0.0743** | **0.0733** | **0.1025** | 0.0004 | 0.3556 | 0.0000 |
| Collaborative Filtering (Item-Based) | 0.0378 | 0.0557 | 0.0594 | **0.2520** | **0.9700** | 0.0054 |
| Content-Based (TF-IDF) | 0.0025 | 0.0020 | 0.0018 | 0.0160 | 0.4183 | **0.8455** |
| Context-Aware (Heuristic) | 0.0741 | 0.0739 | 0.0980 | 0.0004 | 0.1200 | 0.1000 |

The non-personalized baseline achieves the highest overlap accuracy precisely because grocery demand is concentrated in a small set of staples. Collaborative filtering sacrifices headline precision for substantially higher coverage and diversity, reflecting genuine personalization. The TF-IDF content model excels at serendipity, surfacing relevant but non-obvious products. The context-aware heuristic matches the popularity baseline on ranking metrics while adding situational nuance through hour-of-day, day-of-week, and recency signals.

---

## Business Recommendation

No single algorithm is optimal across all dimensions. We recommend a **staged deployment architecture**: use popularity-based scoring for new users (cold-start), layer in item-based personalization once sufficient history exists, and invoke TF-IDF to support basket expansion and discovery goals. Context-aware scoring acts as a tie-breaker and is most impactful for session-specific reranking. This blend maximizes both conversion and long-term customer retention.

---

## Key Challenges & Mitigations

- **Cold-start:** New users receive non-personalized recommendations until sufficient interaction history is available; onboarding signals (e.g., initial category preferences) accelerate profile building.  
- **Popularity bias:** Diversity constraints and occasional injection of long-tail items counteract the feedback loop that concentrates recommendations on a handful of SKUs.

---

## Conclusion

Grocery recommendation is fundamentally a habitual-behavior problem. Simple baselines are hard to beat on pure accuracy, but personalization, diversity, and serendipity deliver measurable business value beyond overlap metrics. The optimal production system combines all four approaches in a layered architecture calibrated to user history depth and business objective.  
