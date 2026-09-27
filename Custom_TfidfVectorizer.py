import re
import array
import gc
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
import os

from hpc_math import (fast_l2_normalize, fast_threshold_and_diag, fast_csr_hstack,
                      fast_filter_csr,
                      fast_char_ngrams, fast_fixed_char_ngrams,
                      fast_char_wb_ngrams, fast_fixed_char_wb_ngrams)

class HybridTfidfVectorizer:
    def __init__(self, max_features=None, analyzer='word', ngram_range=(1, 1), min_df=1):
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df

    def _tokenize(self, text):
        text = str(text).lower()
        if self.analyzer == 'word':
            return re.findall(r'\b[a-zA-Z]{2,}\b', text)
        elif self.analyzer == 'char_wb':
            if self.ngram_range[0] == self.ngram_range[1]:
                return fast_fixed_char_wb_ngrams(text, self.ngram_range[0])
            else:
                return fast_char_wb_ngrams(text, self.ngram_range[0], self.ngram_range[1])
        else:
            if self.ngram_range[0] == self.ngram_range[1]:
                return fast_fixed_char_ngrams(text, self.ngram_range[0])
            else:
                return fast_char_ngrams(text, self.ngram_range[0], self.ngram_range[1])

    def fit_transform(self, raw_documents):
        cols = array.array('i')
        data = array.array('f')
        indptr = array.array('i')
        indptr.append(0)

        vocab = {}
        v_get = vocab.get
        c_append = cols.append
        d_append = data.append

        n_samples = 0

        for doc in raw_documents:
            n_samples += 1
            doc_counts = {}
            for token in self._tokenize(doc):
                idx = v_get(token)
                if idx is None:
                    idx = len(vocab)
                    vocab[token] = idx
                doc_counts[idx] = doc_counts.get(idx, 0) + 1

            for idx, count in doc_counts.items():
                c_append(idx)
                d_append(count)

            indptr.append(len(cols))

        n_features = len(vocab)

        data_np = np.frombuffer(data, dtype=np.float32)
        cols_np = np.frombuffer(cols, dtype=np.int32)
        indptr_np = np.frombuffer(indptr, dtype=np.int32)

        df_array = np.bincount(cols_np, minlength=n_features)
        valid_mask = (df_array >= self.min_df)

        if self.max_features is not None and self.max_features < n_features:
            df_for_sort = df_array.copy()
            df_for_sort[~valid_mask] = 0
            top_indices = np.argsort(df_for_sort)[-self.max_features:]
            valid_mask[:] = False
            top_indices = top_indices[df_for_sort[top_indices] > 0]
            valid_mask[top_indices] = True

        mapping = np.full(n_features, -1, dtype=np.int32)
        mapping[valid_mask] = np.arange(valid_mask.sum(), dtype=np.int32)

        del vocab
        gc.collect()

        nnz = fast_filter_csr(data_np, cols_np, indptr_np, mapping, n_samples)

        data_compact = data_np[:nnz].copy()
        cols_compact = cols_np[:nnz].copy()
        indptr_compact = indptr_np.copy()

        del data, cols, indptr, data_np, cols_np, indptr_np
        gc.collect()

        df_filtered = df_array[valid_mask]
        n_features_filtered = valid_mask.sum()

        X = sp.csr_matrix((data_compact, cols_compact, indptr_compact), shape=(n_samples, n_features_filtered))

        idf_ = np.log((1.0 + n_samples) / (1.0 + df_filtered)) + 1.0
        X.data *= idf_[X.indices].astype(np.float32)

        fast_l2_normalize(X.data, X.indptr, n_samples)
        return X

def _process_combined_batch(start_idx, end_idx, X_combined, threshold):
    batch = X_combined[start_idx:end_idx]

    sim_batch = batch.dot(X_combined.T)

    fast_threshold_and_diag(
        sim_batch.data,
        sim_batch.indices,
        sim_batch.indptr,
        threshold,
        sim_batch.shape[0],
        start_idx
    )
    sim_batch.eliminate_zeros()
    return sim_batch

def _compute_batched_similarities_parallel(titles, descriptions, weight_title, weight_desc, batch_size, threshold):
    vec_title = HybridTfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    X_title = vec_title.fit_transform(titles)

    vec_desc = HybridTfidfVectorizer(max_features=15000, analyzer='char_wb', ngram_range=(3, 3))
    X_desc = vec_desc.fit_transform(descriptions)

    X_title.data *= np.float32(np.sqrt(weight_title))
    X_desc.data *= np.float32(np.sqrt(weight_desc))

    n_samples = X_title.shape[0]
    n_features_title = X_title.shape[1]
    n_features_desc = X_desc.shape[1]

    nnz_c = X_title.nnz + X_desc.nnz
    data_c = np.empty(nnz_c, dtype=np.float32)
    cols_c = np.empty(nnz_c, dtype=np.int32)
    indptr_c = np.empty(n_samples + 1, dtype=np.int32)

    fast_csr_hstack(X_title.data, X_title.indices, X_title.indptr,
                    X_desc.data, X_desc.indices, X_desc.indptr,
                    data_c, cols_c, indptr_c,
                    n_features_title)

    del X_title, X_desc
    gc.collect()

    X_combined = sp.csr_matrix((data_c, cols_c, indptr_c), shape=(n_samples, n_features_title + n_features_desc))

    n_workers = os.cpu_count() or 4
    batch_indices = [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]

    filtered_batches = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_process_combined_batch)(start, end, X_combined, threshold)
        for start, end in batch_indices
    )

    return sp.vstack(filtered_batches)


def _compute_all_similarities(titles, descriptions, weight_title, weight_desc, threshold):
    vec_title = HybridTfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    X_title = vec_title.fit_transform(titles)

    vec_desc = HybridTfidfVectorizer(max_features=15000, analyzer='char_wb', ngram_range=(3, 3))
    X_desc = vec_desc.fit_transform(descriptions)

    X_title.data *= np.float32(np.sqrt(weight_title))
    X_desc.data *= np.float32(np.sqrt(weight_desc))

    n_samples = X_title.shape[0]
    n_features_title = X_title.shape[1]
    n_features_desc = X_desc.shape[1]

    nnz_c = X_title.nnz + X_desc.nnz
    data_c = np.empty(nnz_c, dtype=np.float32)
    cols_c = np.empty(nnz_c, dtype=np.int32)
    indptr_c = np.empty(n_samples + 1, dtype=np.int32)

    fast_csr_hstack(X_title.data, X_title.indices, X_title.indptr,
                    X_desc.data, X_desc.indices, X_desc.indptr,
                    data_c, cols_c, indptr_c,
                    n_features_title)

    del X_title, X_desc
    gc.collect()

    X_combined = sp.csr_matrix((data_c, cols_c, indptr_c), shape=(n_samples, n_features_title + n_features_desc))

    sim_total = X_combined.dot(X_combined.T)

    fast_threshold_and_diag(sim_total.data, sim_total.indices, sim_total.indptr, threshold, sim_total.shape[0])
    sim_total.eliminate_zeros()

    return sim_total

def tfidf_vectorizer(titles, descriptions, algorithm="batched_parallel", weight_title=0.5, weight_desc=0.5,
                     batch_size=1000, threshold=0.4):
    if algorithm == "batched_parallel":
        return _compute_batched_similarities_parallel(titles, descriptions, weight_title, weight_desc, batch_size,
                                                      threshold)
    elif algorithm == "all":
        return _compute_all_similarities(titles, descriptions, weight_title, weight_desc, threshold)
    else:
        raise ValueError(
            "Érvénytelen algoritmus! Kérlek, használd az algorithm='batched_parallel' vagy algorithm='all' paramétert.")
