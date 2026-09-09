from pathlib import Path
from urllib.parse import urlparse
from v19_runtime_contracts import sha

def bind_offline_weights(records, torch):
    """Replace URL resolution, not model code; only declared local bytes load."""
    def offline_load(url, *unused, **kwargs):
        name = kwargs.get('file_name') or Path(urlparse(url).path).name
        if name not in records:
            raise ValueError(f'undeclared Torch checkpoint request: {name}')
        record = records[name]
        path = Path(record['path'])
        if not path.is_file() or sha(path) != record['sha256']:
            raise ValueError(f'offline LightGlue checkpoint missing/changed: {path}')
        return torch.load(path, map_location=kwargs.get('map_location'), weights_only=False)
    torch.hub.load_state_dict_from_url = offline_load
