# preprocess.py
# Placeholder preprocessing functions for TCGA RNA-seq

import numpy as np
import pandas as pd

def log_normalize(df):
    """Log10(1 + TPM) normalization."""
    return np.log10(1 + df)

def max_normalize(df):
    """Normalize each gene to [0,1] range."""
    return df / df.max()

def bin_expression(df, num_bins=64):
    """Convert expression values into bin IDs."""
    bins = np.linspace(0, 1, num_bins + 1)
    return np.digitize(df.values, bins) - 1  # subtract 1 to get [0..num_bins-1]

