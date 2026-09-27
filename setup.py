from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="HPC TF-IDF Vectorizer",
    ext_modules=cythonize(
        "hpc_math.pyx",
        language_level=3, # Cython 3.2.2
    ),
    include_dirs=[np.get_include()]
)
