# tf-idf
This repository contains my implementation of the TF-IDF algorithm.

### HOW TO USE?
```python
        # 1. Saját HPC TF-IDF vektorizáló és batch-hasonlóság feldolgozó importálása
        from Custom_TfidfVectorizer import tfidf_vectorizer

        try:
            # 2. A saját párhuzamosított, memóriatakarékos batch feldolgozónk meghívása
            sim_total = tfidf_vectorizer(
                titles=b_names,
                descriptions=b_addrs,
                algorithm="batched_parallel",
                weight_title=0.5,
                weight_desc=0.5,
                batch_size=1000,
                threshold=TFIDF_THRESHOLD
            )
        except ValueError:
            continue
```
