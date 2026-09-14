"""Reviewed, named numerical problems. New compositions require explicit definitions."""
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor
from optimization.object.pose import GraphConfig

@dataclass(frozen=True)
class StandaloneScene:
    nodes: Sequence[PoseNode]
    times: np.ndarray
    edges: Sequence[RGBEdge]
    anchor: int
    config: GraphConfig = field(default_factory=GraphConfig)
    reprojections: Sequence[RGBReprojectionFactor] = ()
