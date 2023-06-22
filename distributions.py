import numpy as np
from scipy.stats import beta

def get_distribution(num_heads, total_flips):
    return beta.stats(num_heads + 1, total_flips - num_heads + 1, moments='mv')