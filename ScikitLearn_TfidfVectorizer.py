import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from hpc_math import fast_threshold_and_diag
from joblib import Parallel, delayed
import os

def compute_cell12_similarities(titles, descriptions, threshold=0.4):
    vec_name = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    vec_addr = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 3),
        max_features=15000
    )

    tfidf_name = vec_name.fit_transform(titles)
    tfidf_addr = vec_addr.fit_transform(descriptions)

    sim_name = tfidf_name * tfidf_name.T
    sim_addr = tfidf_addr * tfidf_addr.T

    sim_total = (sim_name + sim_addr) / 2.0

    sim_total.data[sim_total.data < threshold] = 0.0
    sim_total.setdiag(0)
    sim_total.eliminate_zeros()

    return sim_total

def _process_sklearn_batch(start_idx, end_idx, tfidf_name, tfidf_addr, threshold):
    batch_name = tfidf_name[start_idx:end_idx]
    batch_addr = tfidf_addr[start_idx:end_idx]

    sim_batch_name = batch_name * tfidf_name.T
    sim_batch_addr = batch_addr * tfidf_addr.T

    sim_batch_total = (sim_batch_name + sim_batch_addr) / 2.0

    fast_threshold_and_diag(
        sim_batch_total.data,
        sim_batch_total.indices,
        sim_batch_total.indptr,
        threshold,
        sim_batch_total.shape[0],
        start_idx
    )

    sim_batch_total.eliminate_zeros()
    return sim_batch_total

def compute_sklearn_batched_similarities(titles, descriptions, batch_size=1000, threshold=0.4):
    vec_name = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    vec_addr = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 3),
        max_features=15000
    )

    tfidf_name = vec_name.fit_transform(titles)
    tfidf_addr = vec_addr.fit_transform(descriptions)

    n_samples = tfidf_name.shape[0]
    n_workers = os.cpu_count() or 4

    batch_indices = [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]

    filtered_batches = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_process_sklearn_batch)(start, end, tfidf_name, tfidf_addr, threshold)
        for start, end in batch_indices
    )

    return sp.vstack(filtered_batches)

def compute_sklearn_sequential_batched_similarities(titles, descriptions, batch_size=1000, threshold=0.4):
    vec_name = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    vec_addr = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 3),
        max_features=15000
    )

    tfidf_name = vec_name.fit_transform(titles)
    tfidf_addr = vec_addr.fit_transform(descriptions)

    n_samples = tfidf_name.shape[0]
    filtered_batches = []
    f_append = filtered_batches.append

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        sim_batch = _process_sklearn_batch(start_idx, end_idx, tfidf_name, tfidf_addr, threshold)
        f_append(sim_batch)

    return sp.vstack(filtered_batches)

def scikit_tfidf_vectorizer(titles, descriptions, algorithm="batched_parallel", batch_size=1000, threshold=0.4):
    if algorithm == "batched_parallel":
        return compute_sklearn_batched_similarities(titles, descriptions, batch_size=batch_size, threshold=threshold)
    elif algorithm == "batched_sequential":
        return compute_sklearn_sequential_batched_similarities(titles, descriptions, batch_size=batch_size, threshold=threshold)
    elif algorithm == "all":
        return compute_cell12_similarities(titles, descriptions, threshold=threshold)
    else:
        raise ValueError(
            "Érvénytelen algoritmus! Kérlek, használd az algorithm='batched_parallel', 'sequential_batched' vagy 'all' paramétert.")
