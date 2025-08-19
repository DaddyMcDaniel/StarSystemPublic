"""
Agent D Terrain Module - T04
============================

Heightfield generation from PCC terrain specifications.
Implements deterministic noise functions and terrain composition.
"""

from .heightfield import HeightField, create_heightfield_from_pcc
from .noise_nodes import NoiseFBM, RidgedMF, DomainWarp

__all__ = ['HeightField', 'create_heightfield_from_pcc', 'NoiseFBM', 'RidgedMF', 'DomainWarp']