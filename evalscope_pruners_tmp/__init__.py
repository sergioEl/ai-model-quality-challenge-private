"""
evalscope_pruners: Custom sampler extensions for evalscope.

Provides:
  - DiscriminabilitySampler: Prune LCB/AA-LCR by marginal discriminative value (Part A).
  - ImageStressSampler: Prune MMMU using visual-stress signals (Part B).
"""

from .discriminability_sampler import DiscriminabilitySampler
from .image_stress_sampler import ImageStressSampler

__all__ = ["DiscriminabilitySampler", "ImageStressSampler"]
