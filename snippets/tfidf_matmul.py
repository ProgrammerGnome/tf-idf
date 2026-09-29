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

@njit
def fast_threshold_and_diag(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    threshold: float,
    n_samples: int,
    start_idx: int = 0,
) -> None:
    for i in range(n_samples):
        global_row = start_idx + i
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            col = indices[j]
            if global_row == col or data[j] < threshold:
                data[j] = 0.0
