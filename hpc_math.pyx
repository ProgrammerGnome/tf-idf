# distutils: language = c++
# cython: language_level=3

# RUN THE SCRIPT: python setup.py build_ext --inplace

cimport cython
from libc.math cimport sqrt

ctypedef fused real:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void fast_l2_normalize(real[:] data, int[:] indptr, int n_samples) noexcept:
    cdef int i, j, start, end
    cdef double sq_sum, norm

    for i in range(n_samples):
        start = indptr[i]
        end = indptr[i+1]
        sq_sum = 0.0

        for j in range(start, end):
            sq_sum += data[j] * data[j]

        norm = sqrt(sq_sum)

        if norm > 0.0:
            for j in range(start, end):
                data[j] /= norm

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fast_threshold_and_diag(real[:] data, int[:] indices, int[:] indptr, double threshold, int n_samples, int start_idx=0) noexcept:
    cdef int i, j, col, global_row
    for i in range(n_samples):
        global_row = start_idx + i
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
            if global_row == col or data[j] < threshold:
                data[j] = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int fast_filter_csr(real[:] data, int[:] cols, int[:] indptr, int[:] mapping, int n_samples) noexcept:
    """Zero-copy mátrix szűrő: helyben tömöríti az adatokat, megkerülve a SciPy borzalmas CSC másolásait!"""
    cdef int i, j, col, new_col
    cdef int write_idx = 0
    cdef int start, end

    for i in range(n_samples):
        start = indptr[i]
        end = indptr[i+1]
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fast_csr_hstack(real[:] data_t, int[:] cols_t, int[:] indptr_t,
                           real[:] data_d, int[:] cols_d, int[:] indptr_d,
                           real[:] data_c, int[:] cols_c, int[:] indptr_c,
                           int offset) noexcept:
    cdef int n_samples = indptr_t.shape[0] - 1
    cdef int i, j, start, end
    cdef int ptr_c = 0

    indptr_c[0] = 0

    for i in range(n_samples):
        start = indptr_t[i]
        end = indptr_t[i+1]
        for j in range(start, end):
            data_c[ptr_c] = data_t[j]
            cols_c[ptr_c] = cols_t[j]
            ptr_c += 1

        start = indptr_d[i]
        end = indptr_d[i+1]
        for j in range(start, end):
            data_c[ptr_c] = data_d[j]
            cols_c[ptr_c] = cols_d[j] + offset
            ptr_c += 1

        indptr_c[i+1] = ptr_c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_char_wb_ngrams(str text, int n_min, int n_max):
    cdef list tokens = []
    cdef str word
    cdef int length, n, i
    for word in text.split():
        word = " " + word + " "
        length = len(word)
        for n in range(n_min, n_max + 1):
            for i in range(length - n + 1):
                tokens.append(word[i:i + n])
    return tokens

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_fixed_char_wb_ngrams(str text, int n):
    cdef list tokens = []
    cdef str word
    cdef int length, i
    for word in text.split():
        word = " " + word + " "
        length = len(word)
        for i in range(length - n + 1):
            tokens.append(word[i:i + n])
    return tokens

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_char_ngrams(str text, int n_min, int n_max):
    cdef int length = len(text)
    cdef int n, i
    cdef list tokens = []
    for n in range(n_min, n_max + 1):
        for i in range(length - n + 1):
            tokens.append(text[i:i + n])
    return tokens

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_fixed_char_ngrams(str text, int n):
    cdef int length = len(text)
    cdef int i
    cdef list tokens = []
    for i in range(length - n + 1):
        tokens.append(text[i:i + n])
    return tokens
