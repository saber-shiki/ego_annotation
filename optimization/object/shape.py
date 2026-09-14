"""Rigid canonical rebinding preserves every world vertex."""
import copy
import numpy as np
from observations.common import pose_from_values
from utils.data import RKEY, TKEY

def rebind_canonical(vertices, rows, Q, b):
    """c_old=Q*c_new+b; preserve all world vertices under a rigid gauge change."""
    Q,t=pose_from_values(Q,b)
    new_vertices=(vertices-t)@Q
    new_rows=copy.deepcopy(rows)
    for old,new in zip(rows,new_rows):
        r,p=pose_from_values(old[RKEY],old[TKEY])
        new[RKEY]=(r@Q).tolist();new[TKEY]=(p+r@t).tolist()
        if not np.allclose(new_vertices@(r@Q).T+p+r@t,vertices@r.T+p,atol=1e-8):
            raise ValueError('canonical gauge broke world geometry')
    return new_vertices,new_rows
