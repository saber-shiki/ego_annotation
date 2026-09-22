"""Typed observation features only; no policy, action, reward or display mutation."""
from dataclasses import dataclass
import numpy as np
from data.immutable import frozen_array
from data.timeline import timeline_ids
from observations.interaction.adapter import SIDES, DIGITS


@dataclass(frozen=True)
class InteractionTimeline:
    frame_idx: np.ndarray
    contact_state: np.ndarray
    contact_prior: np.ndarray
    prior_available: np.ndarray
    primary_declared: np.ndarray
    primary_known: np.ndarray
    visibility: np.ndarray
    primary_visual_eligible: np.ndarray
    # This is just an observation-side eligibility mask, NOT final display validity.
    display_observation_eligible: np.ndarray


def compile_interaction_timeline(evidence, frame_ids, object_id):
    ids = timeline_ids(frame_ids)
    n = len(ids)
    contact = np.full((n,2), 'unresolved', dtype='<U20')
    prior = np.zeros((n,2), np.float64); available = np.zeros((n,2), bool)
    primary = np.zeros((n,2,5),bool); known = primary.copy()
    visibility = np.full((n,3),'unresolved',dtype='<U20')
    for i,frame in enumerate(ids):
        for j,entity in enumerate((object_id,*SIDES)):
            visibility[i,j] = evidence.visibility[frame,entity].state
        for j,side in enumerate(SIDES):
            row = evidence.declarations[frame,side,object_id]
            contact[i,j] = row.contact_state
            if row.prior_probability is not None:
                prior[i,j] = row.prior_probability;available[i,j] = True
            for k,pad in enumerate(row.pads):
                if pad.digit != DIGITS[k]:raise ValueError('anatomical digit ordering changed')
                known[i,j,k] = pad.declared is not None
                primary[i,j,k] = pad.declared is True
    visible = np.isin(visibility, ['visible','partially_visible'])
    eligible = primary & (contact == 'likely_contact')[:,:,None] & visible[:,0,None,None] & visible[:,1:,None]
    # Geometry validity, source MANO availability and reachability are separate.
    return InteractionTimeline(*(frozen_array(a) for a in (
        ids,contact,prior,available,primary,known,visibility,eligible,visibility!='out_of_frame')))
