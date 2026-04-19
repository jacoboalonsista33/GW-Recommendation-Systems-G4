# Final Project: Grocery Purchases Using Instacart Data

**Course:** Recommendation Engines  
**Professor:** Ignacio de Córdoba Álvaro  
**University:** IE University  
**Group Members:** Jacobo Galindo, Javier López-Usero, Samer Roz, Pablo Torres & Paula Candiles

---

## 1\. Domain Analysis & Data Description

Online grocery shopping has been on the rise since the digitalization of many daily activities. Online retailing companies have created a new market that supplies products not just for their local clients, but also for national and international markets, covering necessities for millions of people worldwide. It is here where online platforms like Instacart make their business, operating on the web and delivering products straight to your door for many companies such as Costco, Walmart, and other giants of the industry, which is both convenient and cheap. In this market, understanding customer preferences, behaviours, and purchasing patterns is the highest driving factor of profit, since it allows for a higher optimization of resources such as time, energy, or advertising expenses. Business value lies in recommendations that hit what the customer wants, create new needs for them, and suggest innovative products that open new markets, proving key not only accuracy but also the likes of serendipity, novelty or diversity. That is why recommendation systems are key to drive business growth and optimization of those proves much more important than just mere profit, as it results in loyalty and satisfaction.

The InstaCart Online Grocery Basket dataset was released by InstaCart in 2017 as part of the Kaggle Market Basket Analysis competition, and provides anonymized data from real InstaCart orders. The files include 6 relational files that are connected by primary keys through the IDs, and the description of that connection. The most important variables and hence, those we will focus our analysis on are: users, products, orders, aisles, and departments. Overall, there are 3.4 million orders, 206 thousand users, and 49 thousand products, making this dataset considerably large. The information provided includes behavioral data on the customers, such as order frequency, time of order, and days between orders. It also includes product data like names, categories, aisles, and reordering history, which is key for analyzing purchasing patterns. This mix allows for a thorough analysis of the relationship between user behaviour and product characteristics, providing valuable insights and real-life situations.

The problem that we will face is the prediction of user behaviour in different situations, and what the recommendation systems can do to increase the quality of the user recommendations. It will include binary classification problems, such as the possibility of reordering a certain product based on historical data, or others, such as the balance between recommending the most popular products or better fits. The final aim is to determine which system drives the most value both for the business and the customer, and since value is driven by quality, which is the best recommendation system overall for InstaCart’s purposes and operations.

Passing over to the quality of the data, the first point worth mentioning is missing values. In the whole dataset, there is no lack of information coming from missing values, since the dataset was specifically selected to be complete in all the fields of any order in the different datasets, and includes all relevant data for the competition. However, there are missing values in the days\_since\_prior\_order column, which is logical since there will be many products that have been ordered for the first time, hence they don’t need that attribute. It is not a lack of information; it is simply a way to portray an attribute. Therefore, it can be said that the dataset’s quality is excellent and appropriate for analysis and usage, and the results will be of equal value since they are being done with all possible information and data quality couldn’t possibly be better.

Regarding the users, there are exactly 206.209 unique users, which matches perfectly with the number of NaNs in the days\_since\_prior\_order column, indicating that every user has ordered for the first time once and there are no discrepancies in this aspect. This many users provide a big enough sample to test a system effectively. Users average 10 products per order and 16.2 orders per user, which indicates small but regular purchases, since the majority of orders take place between 5 and 11 days after the last one. This data shows that regular and daily data is reflected on the dataset, since this pattern reveals that in most cases, the weekly purchase is done through this platform, with a spike of 300,000 people expressing such behaviors and a 35% of the whole customer base buying on Saturday and Sunday. This aspect is key for interpretability and generalization, since a representative sample allows for knowledge acquisition and testing for real-life scenarios. Unique products per user average 67, which ensures variety and that many of the basic needs are being covered, not just specific aspects such as meat, vegetable shopping or pet products. Users average 16 orders, ensuring many iterations of weekly shopping and, therefore, representative information over time, being affected by all the factors that will affect any system that wants to provide such recommendations. Lastly, all products have been ordered in a large enough number, with a mean of 680 orders per product and a left-skewed distribution, and only 3 that have never been ordered, requiring a cold start.

Overall data quality is high, and it will provide clean, accurate, and representative insights for the purpose of this project.

## 2\. Data Preprocessing & Feature Engineering

The dataset used in this project corresponds to the Instacart Online Grocery Basket Analysis, which contains detailed information about user purchase behavior, patterns, and decision-making. Using the prior data, it includes over 3.1 million interactions, 38,885 users, and more than 43,000 products, resulting in a highly sparse user–item interaction space (approximately 99.82% sparsity). This sparsity is normal for big-data projects since some popular products get all the attention and millions of them are ordered fewer times. This sparsity doesn’t imply any lack of data, and trying to treat it with the mean, for example, would lead to massive errors, so it will be left as it is.

To construct a unified dataset suitable for recommendation tasks, several data integration steps were required. First, the products table was enriched by merging it with aisle and department information, allowing each product to be associated with its corresponding category for the sake of simplicity. Then, transactional data from prior orders was merged with the orders table and the enriched product dataset in order to have all valuable information stored together and for each record to be complete. This resulted in a comprehensive interaction table containing user identifiers, product identifiers, and contextual variables such as order number, day of the week, and reorder flag for each of the orders made.

Although this notebook focuses on non-personalized models, several implicit features were engineered to support the recommendation process. The number of times each product was purchased was used as a proxy for product popularity. Additionally, the reorder rate—computed as the average value of the reordered flag—was used to capture habitual purchasing behavior. These features are particularly relevant in the grocery domain, where users frequently repurchase the same items and meaningful conclusions are drawn from them.

Exploratory analysis provided important insights into the structure of the dataset. Product popularity follows a clear long-tail distribution, where a small number of products account for a large proportion of total purchases. At the same time, the global reorder rate is approximately 59%, indicating that repeated purchases are a dominant pattern, very common in regular purchasing. User activity is also highly variable, with some users making only a few purchases and others exhibiting very high levels of engagement.

To ensure a realistic evaluation setup, a temporal train/test split was implemented. For each user, the last observed order was used as the test set, while all previous orders were used for training. This approach simulates a real-world recommendation scenario in which the system predicts the next purchase based on past behavior by using all the information available to make decisions and check if they are correct or not with the real decisions made.

To maintain consistency in evaluation, only users present in both training and test sets were retained, meaning that users with only one purchase are not useful in this analysis. This avoids evaluating on cold-start users and ensures that each user has sufficient historical data. The final split resulted in approximately 550,000 training interactions and 222,000 test interactions, covering around 21,000 users. The test set was then transformed into a ground truth structure, mapping each user to the set of products in their last order, enabling the use of ranking-based evaluation metrics.

Overall, the preprocessing pipeline ensures data consistency, meaningful feature representation, and a realistic evaluation framework for the recommendation models.

## 3\. Non-Personalized Recommender

Non-personalized recommendation models were implemented as baseline approaches to establish a reference point for evaluating more advanced techniques. These models do not incorporate individual user preferences but instead rely on global patterns observed in the dataset.

Three different non-personalized strategies were developed. The first approach, Most Popular, recommends the products with the highest number of purchases in the training set. Although it falls into a feedback loop, this model captures overall demand trends and is commonly used as a simple yet effective benchmark in recommendation systems. The second approach, Top Reorder Rate, focuses on identifying products that are frequently repurchased. Instead of relying solely on popularity, this model combines the reorder rate with the number of purchases using logarithmic scaling, providing a more nuanced signal. This allows the model to prioritize products that are both commonly bought and consistently reordered, reflecting habitual consumption patterns. The third approach, Weighted Reorder Score, refines this idea further using a Bayesian framework. It computes a weighted score that balances each product's individual reorder rate against a global average, controlled by a minimum purchase threshold set at the 90th percentile of purchase counts. This shrinks estimates for rarely purchased products toward the global mean, producing more reliable rankings and reducing the influence of low-frequency items.

All models were evaluated using a comprehensive set of metrics. Precision@10 measures the proportion of recommended items that appear in the user's actual next order, while Recall@10 measures the proportion of relevant items successfully retrieved within the top 10 recommendations. NDCG@10 accounts for the ranking order of correct predictions. Beyond accuracy, coverage measures the proportion of the product catalog that appears in recommendations, diversity measures how varied the recommended items are in terms of aisle and department, and serendipity captures how many relevant but non-obvious items are surfaced. Additionally, RMSE and MAE were computed by treating each model's scores as predicted reorder probabilities against actual reorder labels. Evaluation was performed across all users in the test set, using their last order as ground truth.

The results show that the Most Popular and Top Reorder Rate approaches achieve comparable ranking performance, while the Weighted Reorder Score provides a more calibrated signal by dampening noise from infrequent products. Since all three models recommend the same fixed list to every user, coverage is identical and minimal across all approaches. These results are consistent with expectations given the characteristics of the dataset. The nature of grocery shopping behavior can explain the relatively strong performance of popularity-based models: a small number of products dominate overall consumption, and many users tend to purchase similar staple items. As a result, recommending globally popular products already captures a significant portion of user demand.

However, these models present important limitations. Since they do not account for individual user preferences, they generate the same recommendations for all users. This lack of personalization prevents them from capturing niche interests or adapting to user-specific behavior. Additionally, these approaches reinforce popularity bias, reducing the diversity of recommendations and causing the same small set of products to be recommended repeatedly, as confirmed by the near-zero coverage scores.

Despite these limitations, non-personalized recommenders provide a strong and necessary baseline. They establish a reference point for performance and highlight the need for more advanced methods that incorporate user-level information. This motivates the use of collaborative filtering techniques in subsequent stages of the project, where personalization becomes a key component of the recommendation process.

## 4\. Collaborative Filtering Recommender

The collaborative filtering approach implemented in this project is based on an item-based similarity framework, designed to capture personalized user preferences from historical purchase behavior. Unlike non-personalized methods that rely on global popularity signals, this model aims to recommend products by identifying relationships between items that are frequently co-purchased across users.

The dataset used corresponds to the Instacart online grocery platform, which contains implicit feedback in the form of user–product interactions. Since no explicit ratings are available, interactions are treated as binary signals, where a purchase indicates positive preference. This setup is consistent with real-world recommender systems in e-commerce, where implicit feedback is the dominant signal.

To construct the collaborative filtering model, a user–item interaction matrix was built using a sparse binary representation to ensure computational efficiency. Each row represents a user, each column represents a product, and entries indicate whether a user has previously purchased a given item. Given the large scale and high sparsity of the dataset, the product catalog was reduced to the top 1,500 most frequently purchased items. This filtering step significantly improves tractability while retaining the most informative interactions. Item vectors were subsequently L2-normalized to enable efficient cosine similarity computation directly via matrix multiplication.

The model operates by computing cosine similarity between items based on their co-occurrence patterns across users. For a given user, recommendations are generated by iterating over their purchase history and aggregating similarity scores from the top 30 most similar items for each previously purchased product. Items already in the user's history are excluded to ensure novelty, and final recommendations are ranked by their accumulated score.

To evaluate the model, a temporal train/test split was applied, where the last order of each user serves as the test set and all previous interactions form the training set. This reflects a realistic recommendation scenario where future behavior is predicted from past interactions. Evaluation was conducted using a sample of 500 users and covered ranking metrics (Precision@10, Recall@10, NDCG@10), as well as beyond-accuracy metrics including coverage, diversity, and serendipity.

The item-based collaborative filtering model achieved a Precision@10 of 0.0378 and a Recall@10 of 0.0557 on the offline evaluation sample. While these results remain lower than those obtained with non-personalized baselines on headline overlap metrics, they are consistent with the characteristics of the dataset. The Instacart data exhibits extreme sparsity and relies entirely on implicit feedback, both of which make similarity-based personalization more challenging. Furthermore, grocery shopping behavior is strongly driven by staple products purchased repeatedly, which reduces the relative advantage of personalization and helps explain the strong performance of popularity-based methods. On the other hand, the item-based model is expected to achieve higher coverage and serendipity than the non-personalized approaches, since it generates user-specific recommendation lists rather than a single global ranking.

Overall, this approach demonstrates the ability to introduce personalization into the recommendation process, capturing user-specific preferences beyond global trends. However, it also highlights the limitations of basic item-based collaborative filtering in highly sparse, implicit-feedback environments. These findings motivate the exploration of more advanced techniques in subsequent models, including feature-based and context-aware approaches.

## 5\. Content-Based Recommender

Using product attributes and user interactions, a content-based recommendation model was created to offer customized grocery recommendations. In contrast to non-personalized methods, this approach concentrates on finding product similarities and suggesting products that are closely related to those that each user has previously purchased.

Instead of using a predictive classification model, the method is based on a similarity framework. The dataset was first preprocessed by creating a single textual representation of relevant product data, such as departments, aisles, and product names. The TF-IDF (Term Frequency–Inverse Document Frequency) technique was then used to convert this text data into numerical vectors, highlighting the most significant and unique terms for each product.

The similarity between items is measured using cosine similarity after products are represented as vectors. This makes it possible for the system to recognize products with comparable features. Recommendations are produced by choosing products that are most similar to each user's profile, which is created based on their past purchases.

The model avoids creating a complete user-item interaction matrix in order to guarantee computational efficiency, particularly in light of the numerous products. Rather, only the most important candidates are taken into account when making recommendations, and similarity between product vectors is calculated. This method preserves meaningful recommendation quality while using less memory.

Ranking and beyond-accuracy metrics were used to assess the model. The results indicate limited accuracy in retrieving and ranking relevant items, with low precision (0.0025), recall (0.002), and a low NDCG score (0.0018). Nonetheless, the system achieves high serendipity (0.8455) and moderate diversity (0.4183), indicating that it is successful in suggesting new and less evident products. Only a small percentage of the product catalogue is revealed through recommendations, as coverage is still low (0.016). In general, the model prioritizes innovation and discovery over rigorous accuracy.

The content-based approach provides a straightforward, comprehensible, and scalable solution for recommendation systems in spite of these drawbacks. Before putting more sophisticated models into practice, it acts as a reliable starting point and a useful first step toward personalization.

## 6\. Context-Aware Recommender

A context aware recommender system was implemented to incorporate the effect of the situation in which a purchase takes place. In this project, context is defined using temporal and behavioral variables available in the dataset: day of week, time of day, and recency since the previous order. These variables were transformed into categorical features, grouping hours into periods of the day and recency into frequency-based buckets, allowing the model to distinguish between different purchasing situations.

Given the scale and sparsity of the dataset, a lightweight approach based on context dependent popularity was adopted. Three context-specific signals were constructed: hour-of-day popularity, day-of-week popularity, and recency-based popularity. For each of these dimensions, products were ranked using a score that combines purchase frequency and reorder behavior, prioritizing items that are both frequently purchased and consistently reordered.

The final recommendation score is computed as a weighted combination of the three contextual signals, assigning slightly higher importance to time-of-day patterns while maintaining a balanced contribution from day-of-week and recency effects. For a given user, the model extracts the context of their most recent order, retrieves the relevant products associated with each context, aggregates their scores, and returns the Top-K recommendations.

The results show that the recommendations are largely composed of frequently purchased grocery staples such as bananas, milk, and fresh produce. This reflects the strong role of habitual consumption in the dataset, where a small number of products dominate overall demand. Contextual information helps refine these recommendations by capturing when certain products are more likely to be relevant, although popularity remains the main driving factor.

A limitation of this approach is the lack of user-level personalization, as recommendations are determined by context and global patterns. As a result, users sharing similar contexts may receive similar outputs, reducing diversity. Additionally, only the temporal context is considered, excluding other potentially relevant factors. Despite these limitations, the model provides a scalable and interpretable way to incorporate situational information into the recommendation process, which will be assessed in the comparative evaluation.

## 7\. Comparative Evaluation

To figure out the best recommendation strategy for Instacart, we evaluated our four models: Non-Personalized, Collaborative Filtering, Content-Based (TF‑IDF), and Context-Aware. Training pipelines differ slightly by notebook because of memory limits, but every model is summarized in **`05_evaluation.ipynb`** under the protocol below. Offline evaluation balances overlap-style accuracy on the held-out order with catalogue and diversity signals that proxy business usefulness.

### 7.1 Standardized Metrics Comparison

To avoid comparing each recommender in isolation, we consolidated offline figures in **`05_evaluation.ipynb`**. Every row uses the **same metric labels** so the table is readable in one glance: **Precision@10**, **Recall@10**, and **NDCG** use **K \= 10**; **coverage** is how much of the training catalog appears across top-K recommendations for the evaluated users; **diversity** and **serendipity** follow the definitions encoded in the respective notebooks. **RMSE** and **MAE** only appear where a model actually defines a comparable numeric target—global reorder scores for **Most Popular**, and cosine-based scores checked against binary relevance on the shortlist for **TF‑IDF**. Item-based collaborative filtering is strictly a ranking model on implicit purchases, so those two cells stay empty rather than inventing a pseudo rating loss.

Rows for the non-personalized baseline, collaborative filtering, and TF‑IDF content model are copied from **`01_non_personalized.ipynb`**, **`02_collaborative_filtering.ipynb`**, and **`03_content_based.ipynb`**. The **context-aware** row reuses the hour / day-of-week / recency heuristic from **`04_context_aware.ipynb`**, but the metrics themselves are recomputed inside **`05_evaluation.ipynb`** so the same train-style split and random seed feed one evaluator. Instacart files are read from the working directory when you upload them; if they are missing, **`kagglehub`** downloads **`yasserh/instacart-online-grocery-basket-analysis-dataset`** from Kaggle and points the notebook at the extracted CSVs.

| Approach | RMSE | MAE | Precision@10 | Recall@10 | NDCG | Coverage | Diversity | Serendipity | Context |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Non-Personalized (Most Popular) | 0.7710 | 0.6359 | 0.0743 | 0.0733 | 0.1025 | 0.0004 | 0.3556 | 0.0000 | No |
| Collaborative Filtering (Item-Based) | — | — | 0.0378 | 0.0557 | 0.0594 | 0.2520 | 0.9700 | 0.0054 | No |
| Content-Based (TF-IDF) | 0.5185 | 0.5181 | 0.0025 | 0.0020 | 0.0018 | 0.0160 | 0.4183 | 0.8455 | No |
| Context-Aware (Heuristic) | — | — | 0.0741 | 0.0739 | 0.0980 | 0.0004 | 0.1200 | 0.1000 | Yes |

**Notes.** “—” means there is no meaningful RMSE/MAE line-up across models for that approach. The context-aware numbers in the table are **representative consolidated outputs** from **`05_evaluation.ipynb`** (500 sampled users, random seed 42, identical heuristic weights as notebook `04`). If your local run prints slightly different decimals, replace this row with the dataframe your notebook emits—ranking metrics usually track the popularity baseline closely because staples still crowd the top of the list.

The content-based path is **TF‑IDF over product name, aisle, and department**, scored with cosine similarity against a user profile built from prior purchases. We **do not** report Hit Rate@5 or ROC-AUC for that row anymore; those belonged to an older logistic-regression experiment and would not sit next to the other models fairly.

### 7.2 Accuracy & Ranking Performance

The grocery space presents a unique data challenge because it is heavily driven by people buying the same staple items over and over.

Our non-personalized baseline, which just suggests the most popular products, sets a strong benchmark on overlap-based ranking metrics. A massive chunk of user baskets consists of universal favorites like bananas, organic strawberries, and milk. Because of this, simply recommending the global top sellers captures a ton of user demand right away.

Item-based collaborative filtering still trails that baseline on Precision@10 / Recall@10 in our logged evaluation, even after refreshing the metrics on the latest notebook build. That gap comes from the extreme emptiness of the user–item matrix—roughly **99.8%** sparse—and from implicit feedback only (binary buys instead of graded ratings). Similarity-driven personalization adds variance before it adds precision when every basket looks alike.

The TF‑IDF content model deliberately trades overlap accuracy for exploration: Precision@10 and Recall@10 stay low, yet **serendipity** jumps sharply because recommendations can surface relevant substitutes outside the sheer popularity tail. RMSE and MAE on that row reflect **score calibration against binary relevance on the recommended items**, not a reorder classifier.

The heuristic context-aware recommender blends hourly, weekday, and recency popularity tables. Offline, its Precision@10 and Recall@10 sit essentially next to the Most Popular baseline—exactly what we observe when contextual reranking rarely kicks niche SKUs into a length-10 slate dominated by staples. Context still matters for tie-breaking and for sessions where baskets skew away from the global profile, even when headline overlap metrics barely move.

### 7.3 Business Metrics Comparison

While accuracy is key, business value really comes down to how these models behave in the real world:

**Diversity & Coverage.** The non-personalized model concentrates on the same handful of SKUs for everyone; coverage stays microscopic even though precision looks flattering. Collaborative filtering expands coverage substantially because each user pulls a different neighborhood of items through the filtered catalog. TF‑IDF lands in the middle on coverage but pushes diversity and serendipity upward—useful when the product goal is discovery or basket expansion rather than repeating yesterday’s staples only.

**Computational Efficiency.** Collaborative filtering stayed the heaviest pipeline because we trimmed the catalog to the top 1,500 products and materialized sparse cosine stacks even after those cuts. TF‑IDF profiles and the context-aware scorer stay comparatively light: mostly vector lookups, weighted sums, and dictionary merges over pre-aggregated popularity tables—much easier to imagine inside an online serving path than rebuilding massive similarity graphs every request.

### 7.4 Simulated A/B Testing Strategy

To validate offline findings in a production-like setting, we propose a simulated A/B test comparing the non-personalized baseline against the item-based collaborative filtering model, as these two represent the clearest trade-off between simplicity and personalization observed in our evaluation.

**Hypothesis:** Collaborative filtering will increase average basket size and return rate compared to the Most Popular baseline, despite lower offline Precision@10, because it surfaces more relevant and personalized items beyond universal staples.

**Setup:**

* **Control group (A):** 50% of users receive the Most Popular non-personalized recommendations.  
* **Treatment group (B):** 50% of users receive item-based collaborative filtering recommendations.  
* Users are split randomly but stratified by order frequency (low / medium / high) to ensure both groups are behaviorally comparable.  
* **Duration:** 4 weeks — long enough to capture at least two full purchase cycles per user, given the observed average order gap of 7–10 days.

**Primary online metrics to track:**

* Click-through rate (CTR) on recommended items  
* Add-to-cart rate from recommendations  
* Average basket size per session  
* 4-week return rate (customer retention)

**Secondary metrics:**

* Catalog coverage across recommendations (to detect popularity bias regression)  
* Serendipity proxy: proportion of recommended items outside the user's prior purchase history

**Success criteria:** The treatment group shows a statistically significant lift (p \< 0.05) in at least two of the four primary metrics, with no degradation in return rate.

**Guardrails:** If CTR in group B drops more than 10% below group A within the first week, the test is paused and the non-personalized baseline is restored to avoid harming the user experience.

## 8\. Business Case & Deployment Design

### 8.1 Business Case

For online grocery platforms, recommendation systems directly enhances customer satisfaction while generating quantifiable revenue growth and operational effectiveness.

According to Instacart's data, users typically place 16+ orders with about 10 items in each basket, and a reorder rate of about 59% indicates strong habitual consumption patterns. This makes it easier for recommendation systems to predict demand and automate recurring purchases.

Personalization has been shown to boost performance. Effective recommendation systems can boost average order value (AOV) by 5–15% and conversion rates by 10–30%, according to industry benchmarks. This is mainly because they present complementary and relevant products at the appropriate time. 

Recommendation systems are also essential for navigating the long-tail distribution, which balances high-demand staples with higher-margin or less visible items in a catalog with over 40,000 products and millions of interactions. This increases supplier exposure, inventory turnover, and basket expansion.

Strategically, recommendation engines also enable sophisticated marketing features like targeted product placement, cross-selling, and personalized promotions, which have been demonstrated to increase campaign efficacy by two to three times when compared to generic campaigns.

Recommendation systems, which enable scalable personalization, enhance unit economics, and provide a sustained competitive advantage in a data-driven retail environment, are more than just a feature but the key to growth.

### 8.2 Deployment Design

When it comes to deployment, the system can be added to the Instacart platform through a real-time recommendation pipeline.

1. **Training** \- The first step is to train the model offline, using historical data to figure out features and train the recommendation model.  
2. **Scoring** \- Once the system is set up, it makes predictions for each user by scoring potential products and picking the Top-K items that are most likely to be bought.  
3. **Updates** \- You can update these recommendations every so often (like once a day or once a week) to find the right balance between accuracy and cost.

The deployment architecture should use distributed data processing and storage systems to make it scalable, since the dataset is large and sparse. Precomputing user and product features and limiting candidate sets makes sure that inference works well in production. Also, monitoring systems should be put in place to keep an eye on performance over time using both online and offline metrics (for example, Precision@K / NDCG offline and click-through or conversion online).

Overall, the suggested recommendation system is a practical and scalable solution that fits with business goals, boosts user engagement, and adds measurable economic value.

## 9\. Cold-Start & Bias Mitigation Strategy

### 9.1 Cold-Start Problem

The cold-start problem, which occurs when there is insufficient historical data for new users or new products, is one of the primary difficulties in recommendation systems. It is challenging to provide tailored recommendations for new users since the system does not have past interactions to deduce preferences. In order to solve this, the model can first rely on non-personalized strategies, like suggesting products that are well-liked throughout the world or have a high reorder rate, guaranteeing that the recommendations made during the initial interaction are reasonable.

Additionally, a user profile can be swiftly created using simple contextual or onboarding data (such as initial product choices or preferences). Even in the absence of past purchase data, the system can recommend new products based on how similar they are to current items, thanks to content-based features like department and aisle information.

### 9.2 Bias Mitigation Strategy

Bias, especially popularity bias, which is prevalent in recommendation systems, is another significant issue. Because a few products make up the majority of purchases in the dataset, and models tend to overrecommend these well-liked items, which reduces diversity and limits the exposure of less common products.

Techniques like filtering excessively dominant items, rearranging recommendations to include a mix of popular and less popular products, or implementing diversity constraints can be used to solve this.

Additionally, there is a chance that user behavior patterns will be reinforced, leading to a feedback loop where the system consistently suggests the same kinds of goods. Diversifying the user experience and discovering new preferences can be achieved by implementing exploration mechanisms, such as periodically suggesting new or infrequent items.

In conclusion, resolving bias and cold-start problems is crucial to guaranteeing the recommendation system's long-term success, diversity, and fairness.

## 10\. Conclusions

The goal of this project was to design, build, and evaluate a complete recommendation pipeline for the Instacart platform. By developing four distinct recommender systems, we gained valuable insights into how user behavior, algorithm performance, and business value all connect in the online grocery space.

Our analysis showed that grocery shopping is incredibly habitual. The catalog is extremely sparse, and the data follows a heavy long-tailed distribution. This means a tiny fraction of products generate the vast majority of purchases. Because of this, simple non-personalized models work as a highly effective and cheap baseline that guarantees relevant suggestions for any user.

However, relying only on popularity limits business growth. To keep customers coming back and to increase their average basket sizes, as we outlined in our business case, personalization is an absolute must. Our collaborative filtering experiment highlighted how hard similarity learning is under extreme sparsity, while the TF‑IDF content layer offers a cheap way to diversify suggestions even when overlap metrics stay modest. The context-aware blend adds situational nuance without pretending to overturn staple-heavy baskets overnight.

Ultimately, the best deployment strategy for Instacart is not just one algorithm, but a staged blend: use popularity or lightweight context scoring where confidence is thin, layer item-based personalization when histories exist, and invoke TF‑IDF when merchandising wants complementary or exploratory fills. Cold-start traffic can still lean on global popularity until enough signals arrive.

## 11\. Individual Contributions

Each team member took primary ownership of specific components of the project while all members participated in the overall discussion, validation, and final report writing.

1. **Paula Candiles:** did the content-based recommender (Section 5), Section 8 (Business Case & Deployment Design), Section 9 (Cold-Start & Bias Mitigation), and Section 10 (Conclusions).  
2. **Jacobo Galindo:** Did the Non-personalised Recommender and Collaborative Filtering Recommender parts.  
3. **Javier López-Usero:** Led data preprocessing and feature engineering (Section 2\) and coordinated the overall evaluation framework in `05_evaluation.ipynb`. Contributed to Section 1 (Domain Analysis).  
4. **Samer Roz:** Led the collaborative filtering recommender (Section 4), including the sparse interaction matrix, item similarity computation, and ranking evaluation.   
5. **Pablo Torres:** Led the non-personalized recommender (Section 3), implementing all three baseline strategies and the Bayesian scoring approach. 

To guarantee consistency and quality throughout all sections, every team member participated in the writing of the final report, the discussion of findings, and the validation of the overall strategy.

## 12\. AI Usage Disclosure

Artificial intelligence tools were used to support the development and documentation of this project. The following describes their usage in accordance with the course AI Usage Policy:

**Tools used:** ChatGPT (OpenAI) and Claude.

**How they were used:**

- *Report writing:* ChatGPT and Claude were used to suggest phrasing and improve the clarity and coherence of written sections. All content, arguments, and conclusions were authored and validated by the team; AI suggestions were edited and adapted to accurately reflect our findings.  
- *Code assistance:* AI was used during development to autocomplete code and to debug errors in the evaluation notebook. Also helped in algorithmic logic, model design decisions, and parameter choices, but all were validated by the team.  
- *Prompts used:* Examples include "improve the clarity of this paragraph about collaborative filtering sparsity," "suggest a way to phrase a cold-start limitation," and "why might this cosine similarity computation return NaN."

All results, feature engineering decisions, modeling choices, and evaluation techniques are entirely the work of the authors. AI-generated content was reviewed, corrected where necessary, and adapted to meet project goals and academic standards. The team takes full responsibility for any errors in the submitted work.  
