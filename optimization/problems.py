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


@dataclass(frozen=True)
class PreparedVisualFit:
    annotations: object
    pose_report: object
    mano_bridge: object
    hawor_params: object
    mano_models: object
    output_dir: object
    object_id: str
    appearance_mask_manifest: object
    interaction_judgments: object
    device: str = 'cpu'
    iterations: int = 240
    render_width: int | None = None

    def __post_init__(self):
        if self.appearance_mask_manifest is None or self.interaction_judgments is None:
            raise ValueError('prepared visual fitting requires appearance and interaction evidence')
        if self.iterations < 1:
            raise ValueError('visual fitting requires positive iterations')
