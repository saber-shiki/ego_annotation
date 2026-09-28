"""Read-only live S3 verification and evaluation extraction. Never relocate/overwrite a bundle."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,datetime
sys.path.insert(0,'/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment')
from observations.s3_cache import client,DEFAULT_CONFIG,sha_file
ROOT=Path(__file__).resolve().parent
s3,bucket=client(DEFAULT_CONFIG)
inv=json.loads((ROOT/'inventory.json').read_text())
packages=[p for p in inv['packages'] if p['case']!='hot3d_milk_823a906_minfix_v1_20260918']
# Reference availability and size for every file, rather than relying on upload receipts.
remote={}
for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix='ego_annotation/intermediate_artifacts/objects/sha256/'):
    remote.update({o['Key']:o['Size'] for o in page.get('Contents',[])})
checks=[]
for p in packages:
    m=json.loads(Path(p['manifest_parsed_path']).read_text())
    bad=[r['path'] for r in m['files'] if remote.get(r['object_key'])!=r['bytes']]
    checks.append({'case':p['case'],'kind':p['kind'],'files':len(m['files']),'missing_or_size_mismatch':bad})
(ROOT/'s3_reference_check.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unique_content_objects_listed':len(remote),'checks':checks},indent=2)+'\n')
if any(c['missing_or_size_mismatch'] for c in checks):raise ValueError('S3 content unavailable')
for p in packages:
    case=p['case'];kind=p['kind'];out=ROOT/case/kind;out.mkdir(parents=True,exist_ok=True)
    m=json.loads(Path(p['manifest_parsed_path']).read_text());records={r['path']:r for r in m['files']};original=Path(p['original_root']);downloaded={}
    def fetch(name):
        rel=Path(name).relative_to(original).as_posix() if Path(name).is_absolute() else name
        row=records[rel];dst=out/rel
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();n=0
            with s3.get_object(Bucket=bucket,Key=row['object_key'])['Body'] as f,dst.open('xb') as g:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b);g.write(b)
            if (h.hexdigest(),n)!=(row['sha256'],row['bytes']):raise ValueError('bad download '+rel)
        elif sha_file(dst)!=row['sha256']:raise ValueError('bad cached copy '+rel)
        downloaded[rel]=dict(row,GET_SHA256_verified=True)
        return dst
    if kind=='preprocessing':
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,records))
    else:
        bundle=json.loads(fetch('source_observations.json').read_text())
        # All source manifests plus annotations/MANO/mesh/RGB arrays; omit large dense depth,
        # which is unnecessary for the independent retained-surface/trajectory audit.
        names=[r['path'] for k,r in bundle['sources'].items() if isinstance(r,dict) and 'sha256' in r and k!='conditioned_depth']
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,names))
        vis=json.loads(fetch(bundle['sources']['visual_geometry_report']['path']).read_text())
        qc=json.loads(fetch(bundle['sources']['visual_native_qc']['path']).read_text())
        for k in ('image','mask'):fetch(qc[k])
        ann=json.loads(fetch(bundle['sources']['annotations']['path']).read_text())
        ids=sorted({0,29,59,89,119,149,int(vis['appearance_anchor_frame_idx'])})
        raw=json.loads(fetch(bundle['sources']['raw_frames']['path']).read_text())
        # discover selected RGB frames directly in package, independent of manifest schema.
        selected=[]
        for rec in raw.get('frames',[]):
            if int(rec.get('frame_idx',-1)) in ids:
                for k,v in rec.items():
                    if isinstance(v,str) and v.endswith(('.png','.jpg','.jpeg')):
                        try:fetch(v)
                        except (ValueError,KeyError):pass
        # Raw video is retained for duration/identity verification, no model use.
        if 'input/raw.mp4' in records:fetch('input/raw.mp4')
        frame_names=set()
        for f in ann['frames']:
            if int(f['frame_idx']) not in ids:continue
            for o in f.get('objects',[]):
                if o.get('object_id')==vis['object_id'] and o.get('mask_path'):frame_names.add(o['mask_path'])
            for k,v in f.items():
                if isinstance(v,str) and v.endswith(('.jpg','.png','.jpeg')):
                    try:
                        rel=Path(v).relative_to(original).as_posix()
                        if rel in records:frame_names.add(v)
                    except ValueError:pass
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,frame_names))
    (ROOT/case/(kind+'_GET_verification.json')).write_text(json.dumps({'manifest_key':p['key'],'source_bundle_sha256':m['binding']['source_bundle_sha256'],'role':'evaluation_byte_copies_not_relocated_source','files':list(downloaded.values())},indent=2)+'\n')
    print(json.dumps({'case':case,'kind':kind,'GET_verified':len(downloaded),'bytes':sum(x['bytes'] for x in downloaded.values())}),flush=True)
