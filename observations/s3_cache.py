"""Content-addressed S3 retention; create-only manifests published after GET SHA256 checks.

No credentials are placed in manifests. All observation JSON and receipts are
preserved byte-for-byte. Their original absolute root is needed when restoring
for pipeline use. New downstream code consumes the frozen bundle, not observe.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

PREFIX = 'ego_annotation/intermediate_artifacts'
DEFAULT_CONFIG = Path.home()/'.config/ego-artifact-s3/credentials.json'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def client(config):
    import boto3
    from botocore.config import Config
    info=config.stat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise PermissionError('S3 configuration must be owner-only')
    cfg=json.loads(config.read_text()); bucket=cfg.pop('bucket')
    if not isinstance(cfg.get('verify'),str) or not Path(cfg['verify']).is_file():
        raise ValueError('an explicit trusted CA file is required; insecure TLS is prohibited')
    return boto3.client('s3',**cfg,config=Config(signature_version='s3v4',
        s3={'addressing_style':'path'},max_pool_connections=12,
        retries={'max_attempts':3,'mode':'standard'},connect_timeout=15,read_timeout=120)),bucket


def missing(exc):
    return getattr(exc,'response',{}).get('Error',{}).get('Code') in ('404','NoSuchKey','NotFound')


def get_hash(s3,bucket,key):
    h=hashlib.sha256(); n=0
    response=s3.get_object(Bucket=bucket,Key=key)
    with response['Body'] as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block); n+=len(block)
    return h.hexdigest(),n


def immutable_put(s3,bucket,key,body,sha,size):
    from botocore.exceptions import ClientError
    try:
        head=s3.head_object(Bucket=bucket,Key=key)
    except ClientError as exc:
        if not missing(exc): raise
        try:
            s3.put_object(Bucket=bucket,Key=key,Body=body,ContentLength=size,
                Metadata={'sha256':sha},IfNoneMatch='*')
        except ClientError as race:
            if race.response['Error']['Code'] not in ('PreconditionFailed','412'): raise
    else:
        if head['ContentLength']!=size:
            raise ValueError(f'conflicting existing object: {key}')
    # Always verify actual remote bytes, including existing objects; ETag is not SHA256.
    if get_hash(s3,bucket,key)!=(sha,size):
        raise ValueError(f'remote content mismatch: {key}')


def discover(root):
    paths=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError(f'symlink outside closure: {p}')
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if rel=='state/recovery/lock': continue
        if p.name.startswith('.astra-credential') or p.name=='credentials.json':
            raise ValueError('credential file must never enter artifact storage')
        paths.append((p,rel))
    return paths


def publish(s3,bucket,root,case,kind,prefix,workers):
    root=Path(root).resolve()
    if not root.is_dir(): raise ValueError('source root missing')
    if kind=='observations':
        b=root/'source_observations.json'
        metadata=json.loads(b.read_text())
        if metadata.get('schema') not in ('ego.annotation.observation_bundle.v1','ego.annotation.observation_bundle.v2'):
            raise ValueError('requires completed observation bundle')
        bind={'source_bundle_sha256':sha_file(b),'source_schema':metadata['schema'],
              'identity':metadata['identity'],'capabilities':metadata.get('capabilities')}
    else:
        b=root/'inspection.json'; metadata=json.loads(b.read_text())
        if (metadata.get('source_unchanged') is not True or metadata.get('missing')
                or metadata.get('status') not in ('complete_preprocessing_readback','complete_with_measurement_gaps')):
            raise ValueError('requires successful independent preprocessing')
        bind={'source_bundle_sha256':metadata['source_manifest_sha256'],
              'source_manifest':metadata['source_manifest'],'status':metadata['status']}
    paths=discover(root)
    def upload(item):
        path,relative=item
        before=path.stat(); sha=sha_file(path)
        key=f'{prefix}/objects/sha256/{sha[:2]}/{sha}'
        with path.open('rb') as f: immutable_put(s3,bucket,key,f,sha,before.st_size)
        after=path.stat()
        if (after.st_mtime_ns,after.st_size)!=(before.st_mtime_ns,before.st_size) or sha_file(path)!=sha:
            raise ValueError(f'source changed during upload: {path}')
        return {'path':relative,'bytes':before.st_size,'sha256':sha,'object_key':key}
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(upload,paths):
            rows.append(row)
            if len(rows)%100==0: print(f'{case}/{kind}: {len(rows)}/{len(paths)} objects verified',flush=True)
    if [r for _,r in discover(root)]!=[r['path'] for r in rows]:
        raise ValueError('source file list changed during publication')
    manifest={'schema':'ego.annotation.s3_artifact_package.v1','case':case,'kind':kind,
        'original_root':str(root),'binding':bind,'files':rows,
        'verification':'every object GET streamed and SHA256 compared before manifest publication',
        'restore_contract':'preserve original absolute paths for observation receipt consumers; no restamping',
        'training_or_policy_executed':False}
    body=encoded(manifest);sha=hashlib.sha256(body).hexdigest()
    key=f'{prefix}/packages/{case}/{kind}/{sha}.json'
    immutable_put(s3,bucket,key,body,sha,len(body))
    return {'s3_uri':f's3://{bucket}/{key}','manifest_key':key,'sha256':sha,
            'kind':kind,'case':case,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),
            'source_bundle_sha256':bind['source_bundle_sha256'],'all_objects_readback_verified':True}


def read_manifest(s3,bucket,key):
    data=s3.get_object(Bucket=bucket,Key=key)['Body'].read()
    if hashlib.sha256(data).hexdigest()!=PurePosixPath(key).stem:
        raise ValueError('manifest filename/content SHA256 mismatch')
    m=json.loads(data)
    if m['schema']!='ego.annotation.s3_artifact_package.v1': raise ValueError('unknown schema')
    for r in m['files']:
        p=PurePosixPath(r['path'])
        if p.is_absolute() or '..' in p.parts: raise ValueError('unsafe path')
        if PurePosixPath(r['object_key']).name!=r['sha256']: raise ValueError('unsafe object key')
    return m


def restore(s3,bucket,key,destination):
    m=read_manifest(s3,bucket,key);destination=Path(destination).resolve()
    if m['kind']=='observations' and str(destination)!=m['original_root']:
        raise ValueError('observation receipts bind absolute paths; restore to original_root only')
    destination.mkdir(parents=True,exist_ok=False)
    for row in m['files']:
        p=destination/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();n=0
        response=s3.get_object(Bucket=bucket,Key=row['object_key'])
        with response['Body'] as src,p.open('xb') as dst:
            for block in iter(lambda:src.read(1024*1024),b''):
                dst.write(block);h.update(block);n+=len(block)
        if (h.hexdigest(),n)!=(row['sha256'],row['bytes']): raise ValueError(f'restore mismatch: {p}')
    return {'restored':str(destination),'file_count':len(m['files'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    sub=p.add_subparsers(dest='command',required=True)
    pub=sub.add_parser('publish');pub.add_argument('--root',type=Path,required=True)
    pub.add_argument('--case',required=True);pub.add_argument('--kind',choices=['observations','preprocessing'],required=True)
    pub.add_argument('--prefix',default=PREFIX);pub.add_argument('--workers',type=int,default=8)
    pub.add_argument('--receipt',type=Path,required=True)
    res=sub.add_parser('restore');res.add_argument('--manifest-key',required=True);res.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();s3,bucket=client(args.config)
    if args.command=='publish':
        if not args.case or '/' in args.case or '..' in args.case: raise ValueError('invalid case')
        if args.receipt.exists(): raise FileExistsError(args.receipt)
        result=publish(s3,bucket,args.root,args.case,args.kind,args.prefix,args.workers)
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x') as f: json.dump(result,f,indent=2)
    else:result=restore(s3,bucket,args.manifest_key,args.destination)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
