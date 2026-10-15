"""Create-only retention of observation runs; no model execution or receipt rewriting.

The original bundle stays the consumer entry point because nested receipts bind
absolute paths. A tar snapshot is a byte-preserving backup, NOT a relocated run.
Restore to the recorded original root before using same-run recovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    rows = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'symlink requires explicit closure, refusing: {p}')
        if p.is_file() and p.relative_to(root).as_posix() != 'state/recovery/lock':
            rows[p.relative_to(root).as_posix()] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    return rows


def verify_tree(root, rows):
    current = inventory(root)
    if current != rows:
        changed = sorted(k for k in current.keys() | rows.keys() if current.get(k) != rows.get(k))
        raise ValueError(f'file inventory changed: {changed[:8]}')


def retain(bundle, output, preprocessing=None, resume_partial=False):
    from data.bundle import EpisodeBundle
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    run = bundle.parent
    if output == run or run in output.parents:
        raise ValueError('retention output must be outside source run')
    # Full bundle hashes are checked before backing up any bytes.
    source = EpisodeBundle.open(bundle)
    prep = Path(preprocessing).resolve() if preprocessing else None
    if prep:
        if output == prep or prep in output.parents:
            raise ValueError('retention output must be outside preprocessing')
        report = json.loads((prep / 'inspection.json').read_text())
        if (report.get('source_manifest_sha256') != digest(bundle)
                or report.get('source_unchanged') is not True
                or report.get('missing')
                or report.get('status') not in ('complete_preprocessing_readback', 'complete_with_measurement_gaps')):
            raise ValueError('preprocessing is not a successful readback of this exact bundle')
    if (output / 'retention.json').exists():
        raise FileExistsError('completed retention is immutable')
    output.mkdir(parents=True, exist_ok=resume_partial)
    rows = inventory(run)
    # Archive only producer artifacts, never follow symlinks outside the run.
    partial = output / 'run.tar.partial'
    archive_path = output / 'run.tar'
    if not archive_path.exists():
        if partial.exists():
            raise FileExistsError('incomplete tar requires a separate explicit recovery')
        with tarfile.open(partial, 'w', dereference=False) as archive:
            for name in rows:
                archive.add(run / name, arcname=name, recursive=False)
        archive_path = partial
    verify_tree(run, rows)
    # Verify the actual archived bytes, not just source hashes.
    with tarfile.open(archive_path, 'r') as archive:
        if set(archive.getnames()) != set(rows):
            raise ValueError('archive file closure mismatch')
        for member in archive:
            h = hashlib.sha256()
            with archive.extractfile(member) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != rows[member.name]['sha256']:
                raise ValueError(f'archive content mismatch: {member.name}')
    if archive_path == partial:
        partial.rename(output / 'run.tar')
    prep_rows = None
    if prep:
        prep_rows = inventory(prep)
        destination = output / 'preprocessing'
        destination.mkdir(exist_ok=resume_partial)
        # SMB may reject chmod/copystat; bytes, not POSIX metadata, are the contract.
        for name, row in prep_rows.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not resume_partial or digest(target) != row['sha256']:
                    raise ValueError(f'conflicting partial preprocessing: {target}')
            else:
                shutil.copyfile(prep / name, target)
        verify_tree(prep, prep_rows)
        verify_tree(destination, prep_rows)
    receipt = {
        'schema': 'ego.annotation.retained_observations.v1',
        'source_bundle': str(bundle), 'source_bundle_sha256': digest(bundle),
        'source_schema': source.manifest['schema'],
        'identity': dict(source.manifest['identity']),
        'original_run_root': str(run),
        'archive': {'path': 'run.tar', 'bytes': (output / 'run.tar').stat().st_size,
                    'sha256': digest(output / 'run.tar')},
        'run_files': rows, 'preprocessing_files': prep_rows,
        'original_preprocessing': str(prep) if prep else None,
        'reuse': 'load original hash-bound source_bundle; do not invoke observe',
        'restore': 'restore archive only to original_run_root; absolute receipts must not be rewritten',
        'model_inference_executed': False,
    }
    (output / 'retention.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify(output):
    output = Path(output)
    receipt = json.loads((output / 'retention.json').read_text())
    archive = output / receipt['archive']['path']
    if digest(archive) != receipt['archive']['sha256']:
        raise ValueError('retained archive hash mismatch')
    if receipt['preprocessing_files'] is not None:
        verify_tree(output / 'preprocessing', receipt['preprocessing_files'])
    return {'verified': True, 'source_bundle': receipt['source_bundle'],
            'run_file_count': len(receipt['run_files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    save = sub.add_parser('retain')
    save.add_argument('--bundle', required=True, type=Path)
    save.add_argument('--output', required=True, type=Path)
    save.add_argument('--preprocessing', type=Path)
    save.add_argument('--resume-partial', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'retain':
        r = retain(args.bundle, args.output, args.preprocessing, args.resume_partial)
        print(json.dumps({'retained': str(args.output), 'run_file_count': len(r['run_files'])}))
    else:
        print(json.dumps(verify(args.output)))


if __name__ == '__main__':
    main()
