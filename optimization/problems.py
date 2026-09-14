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


@dataclass(frozen=True)
class FormalManoInterval:
    """One reviewed interval composition with fixed source camera, shape and mesh."""
    case: str
    object_id: str
    annotations: object
    pose_report: object
    completed_mesh: object
    completion_report: object
    depth_npz: Sequence[object]
    output_dir: object
    wilor_root: object
    mano_left: object
    start_frame: int
    end_frame: int
    profile: str = 'formal'
    device: str = 'cpu'
    factor_reports: Sequence[object] = ()
    hand_ray_prior: object | None = None
    numeric_options: dict = field(default_factory=dict)

    def arguments(self):
        from optimization.hand.fit import formal_interval_arguments
        profiles={'formal':{},'hot3d_shared':{'require_active_full_K_mano_contract':True,'visible_surface_depth_order_term':True,'freeze_translation_without_visible_surface_support':True,'gate_translation_with_visible_surface_support':True,'optimize_object_translation':False},
            'contact_geometry':{'contact_state_geometry_likelihood':True},
            'object_delta':{'optimize_object_translation':True},
            'hand_ray_prior':{'use_hand_ray_shift_prior':True},
            'no_dense_barrier':{'dense_observed_surface_barrier':False},
            'visible_depth_order':{'visible_surface_depth_order_term':True},
            'support_scaled_attraction':{'contact_patch_residual_mode':'support_scaled_attraction'},
            'visibility_weighted':{'visibility_weighted_hand_observation':True},'similarity_mapped_raw':{'zero_surface_mode':'similarity_mapped_raw'}}
        if self.profile not in profiles:
            raise ValueError('unreviewed formal MANO profile')
        if not self.depth_npz or self.end_frame<self.start_frame:
            raise ValueError('formal interval requires depth sources and ordered frame bounds')
        argv=[]
        for key,value in [('case',self.case),('object-id',self.object_id),('annotations',self.annotations),
            ('pose-report',self.pose_report),('completed-mesh',self.completed_mesh),
            ('completion-report',self.completion_report),('output-dir',self.output_dir),
            ('wilor-root',self.wilor_root),('wilor-mano-left',self.mano_left),
            ('start-frame',self.start_frame),('end-frame',self.end_frame),('device',self.device)]:
            argv.extend(['--'+key,str(value)])
        for path in self.depth_npz:argv.extend(['--depth-npz',str(path)])
        for path in self.factor_reports:argv.extend(['--factor-report',str(path)])
        if self.hand_ray_prior is not None:argv.extend(['--hand-depth-repair-graph',str(self.hand_ray_prior)])
        if self.profile=='hand_ray_prior' and self.hand_ray_prior is None:
            raise ValueError('hand-ray profile requires a current ray prior')
        if self.profile in {'visible_depth_order','hot3d_shared'} and not self.factor_reports:
            raise ValueError('depth-order profile requires a visible-surface factor report')
        args=formal_interval_arguments(argv)
        args.optimize_contact_state=True
        for key,value in profiles[self.profile].items():setattr(args,key,value)
        for key,value in self.numeric_options.items():
            default=getattr(args,key,None)
            if type(value) not in (int,float) or type(default) not in (int,float):
                raise ValueError('numeric options cannot change problem composition')
            setattr(args,key,type(default)(value))
        return args
