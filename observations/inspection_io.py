"""Bounded read-only I/O for observation inspection. Never unpickle model state."""
from contextlib import ExitStack
from pathlib import Path
import json
import zipfile
import numpy as np
import cv2
from data.timeline import timeline_ids
from utils.state import sha


class SourceReads:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def path(self, value, expected=None):
        path = Path(value).expanduser()
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'observation input missing: {path}')
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'observation input hash mismatch: {path}')
        if str(path) in self.files and self.files[str(path)] != digest:
            raise ValueError(f'observation input changed while reading: {path}')
        self.files[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.path(path, expected).read_text())

    def image(self, path, expected=None, gray=False):
        path = self.path(path, expected)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'observation image is not decodable: {path}')
        return image

    def npz(self, path, expected=None):
        with np.load(self.path(path, expected), allow_pickle=False) as archive:
            return dict(archive)

    def verify_unchanged(self):
        for path, digest in self.files.items():
            if sha(path) != digest:
                raise ValueError(f'inspection mutated or raced with source input: {path}')


def numeric_summary(values):
    a = np.asarray(values, float).ravel()
    finite = a[np.isfinite(a)]
    return {'count': int(a.size), 'finite_count': int(finite.size),
            'min': float(finite.min()) if finite.size else None,
            'median': float(np.median(finite)) if finite.size else None,
            'max': float(finite.max()) if finite.size else None}


def _header(handle):
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f'unsupported numeric observation NPY header: {version}')
    if fortran or dtype.hasobject or dtype.kind not in 'fiu':
        raise ValueError('inspection requires C-order numeric depth/confidence arrays')
    return shape, dtype


def stream_depth(path):
    """Read every depth/confidence byte once, with only one frame resident.

    Full archive hashes are independently checked by SourceReads/EpisodeBundle.
    Partial depth coverage is allowed and reported rather than filled with zeros.
    """
    with np.load(path, allow_pickle=False) as archive:
        ids = timeline_ids(archive['frame_idx'])
        K = np.asarray(archive['intrinsics_fx_fy_cx_cy'], float)
        size = np.asarray(archive['source_size'])
    if K.shape != (len(ids), 4) or not np.isfinite(K).all() or np.any(K[:, :2] <= 0):
        raise ValueError('invalid depth camera timeline')
    if size.shape != (2,) or size.dtype.kind not in 'iu' or np.any(size <= 0):
        raise ValueError('invalid depth source raster size')
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('duplicate NPZ members')
        streams = {name: stack.enter_context(archive.open(name + '.npy')) for name in ('depth', 'confidence')}
        headers = {name: _header(handle) for name, handle in streams.items()}
        expected_shape = (len(ids), int(size[1]), int(size[0]))
        for name, (shape, dtype) in headers.items():
            if shape != expected_shape:
                raise ValueError(f'{name} raster/frame count does not match declared timeline')
            if int(np.prod(shape[1:])) * dtype.itemsize > 256 * 1024 * 1024:
                raise ValueError('single depth frame exceeds explicit inspection memory bound')
        for position, frame in enumerate(ids):
            arrays = {}
            for name, handle in streams.items():
                shape, dtype = headers[name]
                count = int(np.prod(shape[1:])) * dtype.itemsize
                payload = handle.read(count)
                if len(payload) != count:
                    raise ValueError('truncated depth/confidence frame')
                arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape[1:])
            # Match P09's declared numeric-consumer representation before array hashes.
            yield frame, np.asarray(arrays['depth'],np.float32), np.asarray(arrays['confidence'],np.float32), K[position]
        if any(handle.read(1) for handle in streams.values()):
            raise ValueError('undeclared trailing depth/confidence samples')


def track_rows(payload, frame_ids):
    if not isinstance(payload, dict):
        raise ValueError('mask track must be a frame-ID mapping')
    rows = {}
    for key, row in payload.items():
        try:
            frame = int(key)
        except (ValueError, TypeError) as exc:
            raise ValueError('mask track contains a non-frame key') from exc
        if str(frame) != key or frame in rows or not isinstance(row, dict):
            raise ValueError('mask track frame IDs/rows malformed')
        if type(row.get('visible')) is not bool:
            raise ValueError('mask track requires explicit Boolean visibility')
        rows[frame] = row
    if set(rows) != set(frame_ids):
        raise ValueError('mask track does not cover the explicit source timeline')
    return rows
