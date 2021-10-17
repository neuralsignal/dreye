"""
Check which packages exist
"""

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

try:
    import jax
    JAX = True
except ImportError:
    JAX = False