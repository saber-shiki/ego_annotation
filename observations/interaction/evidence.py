"""Appearance observations are distinct from metric ownership masks."""
from pathlib import Path
from utils.state import sha

def appearance_mask_rows(manifest, annotation_path, object_id, frame_ids, source_size):
    """Validate and bind the raw-appearance plane; never fall back to owned masks."""
    if not isinstance(manifest, dict) or manifest.get("object_id") != object_id:
        raise ValueError("appearance manifest object mismatch")
    if manifest.get("metric_owned_masks_consumed") is not False:
        raise ValueError("appearance manifest does not exclude metric-owned masks")
    if manifest.get("source_annotations_sha256") != sha(annotation_path):
        raise ValueError("appearance manifest annotations identity mismatch")
    rows = manifest.get("frames")
    if not isinstance(rows, list):
        raise ValueError("appearance manifest frames missing")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("object_id") != object_id:
            raise ValueError("invalid appearance manifest row")
        idx = int(row["frame_idx"])
        if idx in indexed:
            raise ValueError(f"duplicate appearance frame {idx}")
        path = row.get("appearance_mask_path")
        if row.get("status") == "available":
            if row.get("visible") is not True:
                raise ValueError(f"invisible appearance mask cannot be available at frame {idx}")
            if not path or sha(path) != row.get("appearance_mask_sha256"):
                raise ValueError(f"appearance mask identity mismatch at frame {idx}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            declared = row.get("source_size_wh")
            if (mask is None or not isinstance(declared, list) or len(declared) != 2
                    or mask.shape != (int(declared[1]), int(declared[0]))
                    or abs(mask.shape[1] / mask.shape[0] - float(source_size[0]) / float(source_size[1])) > 0.01):
                raise ValueError(f"appearance mask dimensions mismatch at frame {idx}")
            indexed[idx] = path if np.any(mask) else None
        elif row.get("status") == "unavailable_not_substituted" and path is None:
            indexed[idx] = None
        else:
            raise ValueError(f"invalid appearance availability at frame {idx}")
    expected = {int(value) for value in frame_ids}
    if set(indexed) != expected:
        raise ValueError("appearance manifest frame coverage mismatch")
    return indexed


import cv2
import numpy as np
