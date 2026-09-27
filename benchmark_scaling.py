import os
import re
import time
import tracemalloc
import numpy as np
from sklearn.datasets import fetch_20newsgroups

from ScikitLearn_TfidfVectorizer import scikit_tfidf_vectorizer
from Custom_TfidfVectorizer import tfidf_vectorizer

def run_benchmark():
    print("20 Newsgroups adathalmaz betöltése...")
    categories = ['sci.space', 'comp.graphics', 'rec.autos', 'talk.politics.mideast']
    raw_data = fetch_20newsgroups(subset='all', categories=categories)

    titles_base, descriptions_base = [], []
    for text in raw_data.data:
        subject_match = re.search(r'^Subject:\s*(.*)$', text, re.MULTILINE)
        titles_base.append(subject_match.group(1).strip() if subject_match else "")
        parts = text.split('\n\n', 1)
        descriptions_base.append(parts[1].strip() if len(parts) > 1 else text.strip())

    repeats = int(np.ceil(30000 / len(titles_base)))
    titles_full = np.tile(titles_base, repeats)
    descriptions_full = np.tile(descriptions_base, repeats)

    threshold_val = 0.4
    batch_size_val = 1000

    #limits = list(range(2500, 25000, 2500))
    limits = list(range(17500, 20000, 2500))

    os.makedirs("./outputs", exist_ok=True)
    benchmark_file = "outputs/benchmark_scaling.txt"

    with open(benchmark_file, "w", encoding="utf-8") as f:
        f.write("Limit,Method,Time_sec,RAM_MB,NNZ\n")

    print("Skálázódási benchmark indítása (Párhuzamosított verzió)...")

    for limit in limits:
        print(f"\n--- Tesztelés {limit} dokumentummal ---")
        titles = titles_full[:limit]
        descriptions = descriptions_full[:limit]

        with open(benchmark_file, "a", encoding="utf-8") as f:
            # # 1. Sklearn All (Egy szálon futó, teljes memóriát igénylő)
            # tracemalloc.start()
            # t0 = time.time()
            # sim = scikit_tfidf_vectorizer(titles, descriptions, threshold=threshold_val)
            # time_sec = time.time() - t0
            # _, peak_mem = tracemalloc.get_traced_memory()
            # tracemalloc.stop()
            # f.write(f"{limit},Sklearn_All,{time_sec:.3f},{peak_mem / 1024 / 1024:.2f},{sim.nnz}\n")
            # print(f"  Sklearn_All:               {time_sec:.2f} mp, {peak_mem / 1024 / 1024:.2f} MB")

            # 2. Sklearn Batched (Többszálas, Joblib parallel)
            tracemalloc.start()
            t0 = time.time()
            # A korábbi módosításunk miatt ez a függvény már alapból a loky backendet használja
            sim = scikit_tfidf_vectorizer(titles, descriptions, algorithm="batched_sequential", batch_size=batch_size_val, threshold=threshold_val)
            time_sec = time.time() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            f.write(f"{limit},Sklearn_Batched_Parallel,{time_sec:.3f},{peak_mem / 1024 / 1024:.2f},{sim.nnz}\n")
            print(f"  Sklearn_Batched_Parallel:  {time_sec:.2f} mp, {peak_mem / 1024 / 1024:.2f} MB")

            # # 3. Custom All (Egy szálon futó, saját kód)
            # tracemalloc.start()
            # t0 = time.time()
            # sim = tfidf_vectorizer(titles, descriptions, algorithm="all", threshold=threshold_val)
            # time_sec = time.time() - t0
            # _, peak_mem = tracemalloc.get_traced_memory()
            # tracemalloc.stop()
            # f.write(f"{limit},Custom_All,{time_sec:.3f},{peak_mem / 1024 / 1024:.2f},{sim.nnz}\n")
            # print(f"  Custom_All:                {time_sec:.2f} mp, {peak_mem / 1024 / 1024:.2f} MB")

            # 4. Custom Batched (Többszálas, Joblib parallel)
            tracemalloc.start()
            t0 = time.time()
            # Itt átállítottuk az algoritmust a 'batched_parallel'-re!
            sim = tfidf_vectorizer(titles, descriptions, algorithm="batched_sequential", batch_size=batch_size_val, threshold=threshold_val)
            time_sec = time.time() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            f.write(f"{limit},Custom_Batched_Parallel,{time_sec:.3f},{peak_mem / 1024 / 1024:.2f},{sim.nnz}\n")
            print(f"  Custom_Batched_Parallel:   {time_sec:.2f} mp, {peak_mem / 1024 / 1024:.2f} MB")

    print(f"\nA skálázódási mérések befejeződtek. Eredmény mentve: {benchmark_file}")


if __name__ == '__main__':
    run_benchmark()
