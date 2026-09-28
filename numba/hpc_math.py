import math
import numpy as np
from numba import njit

# ==========================================
# 1. CSR MÁTRIX MŰVELETEK (NUMBA JIT)
# ==========================================

@njit(fastmath=True)
def fast_l2_normalize(data: np.ndarray, indptr: np.ndarray, n_samples: int) -> None:
    for i in range(n_samples):
        start = indptr[i]
        end = indptr[i + 1]
        sq_sum = 0.0

        for j in range(start, end):
            val = data[j]
            sq_sum += val * val

        norm = math.sqrt(sq_sum)

        if norm > 0.0:
            for j in range(start, end):
                data[j] /= norm


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


@njit
def fast_filter_csr(
    data: np.ndarray,
    cols: np.ndarray,
    indptr: np.ndarray,
    mapping: np.ndarray,
    n_samples: int,
) -> int:
    """Zero-copy mátrix szűrő: helyben tömöríti az adatokat."""
    write_idx = 0
    for i in range(n_samples):
        start = indptr[i]
        end = indptr[i + 1]
        indptr[i] = write_idx
        for j in range(start, end):
            col = cols[j]
            new_col = mapping[col]
            if new_col != -1:
                data[write_idx] = data[j]
                cols[write_idx] = new_col
                write_idx += 1
    indptr[n_samples] = write_idx
    return write_idx


@njit
def fast_csr_hstack(
    data_t: np.ndarray,
    cols_t: np.ndarray,
    indptr_t: np.ndarray,
    data_d: np.ndarray,
    cols_d: np.ndarray,
    indptr_d: np.ndarray,
    data_c: np.ndarray,
    cols_c: np.ndarray,
    indptr_c: np.ndarray,
    offset: int,
) -> None:
    n_samples = indptr_t.shape[0] - 1
    ptr_c = 0
    indptr_c[0] = 0

    for i in range(n_samples):
        start = indptr_t[i]
        end = indptr_t[i + 1]
        for j in range(start, end):
            data_c[ptr_c] = data_t[j]
            cols_c[ptr_c] = cols_t[j]
            ptr_c += 1

        start = indptr_d[i]
        end = indptr_d[i + 1]
        for j in range(start, end):
            data_c[ptr_c] = data_d[j]
            cols_c[ptr_c] = cols_d[j] + offset
            ptr_c += 1

        indptr_c[i + 1] = ptr_c


# ==========================================
# 2. N-GRAM ÉS SZÖVEGFELDOLGOZÁS (CPYTHON)
# ==========================================

def fast_char_wb_ngrams(text: str, n_min: int, n_max: int) -> list:
    tokens = []
    for word in text.split():
        w = f" {word} "
        length = len(w)
        for n in range(n_min, n_max + 1):
            tokens.extend(w[i:i + n] for i in range(length - n + 1))
    return tokens


def fast_fixed_char_wb_ngrams(text: str, n: int) -> list:
    tokens = []
    for word in text.split():
        w = f" {word} "
        length = len(w)
        tokens.extend(w[i:i + n] for i in range(length - n + 1))
    return tokens


def fast_char_ngrams(text: str, n_min: int, n_max: int) -> list:
    length = len(text)
    tokens = []
    for n in range(n_min, n_max + 1):
        tokens.extend(text[i:i + n] for i in range(length - n + 1))
    return tokens


def fast_fixed_char_ngrams(text: str, n: int) -> list:
    length = len(text)
    return [text[i:i + n] for i in range(length - n + 1)]
