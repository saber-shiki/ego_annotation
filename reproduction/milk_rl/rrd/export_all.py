from pathlib import Path
import json,subprocess,re,time
from export_policy_rrd import export,sha
root=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_learning_stage1_1024_v1')
old=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_rl_smoke_minfix_v1')
raw=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/hot3d_milk_823a906_minfix_v1_20260918/input/raw_frame_manifest/manifest.json')
out=root/'rrd';out.mkdir(exist_ok=False)
sources=[('zero',old/'zero/candidate.npz')]+[(f'step_{i:04d}',next((root/f'checkpoints/step_{i:04d}/evaluation.manifest.episodes').glob('*.npz'))) for i in [256,512,768,1024]]
results=[]
cli='/mnt/user-home/kupingxin/ego_skrl/.venv/bin/rerun'
for label,p in sources:
    t=time.monotonic();print('START',label,flush=True)
    rec=export(p,raw,out/f'milk_{label}.rrd',label)
    target=rec['output']
    check=subprocess.run([cli,'rrd','verify',target],text=True,capture_output=True,timeout=120)
    (out/f'{label}.verify.log').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError(check.stdout+check.stderr)
    printed=subprocess.run([cli,'rrd','print',target],text=True,capture_output=True,timeout=120,check=True).stdout
    (out/f'{label}.contents.txt').write_text(printed)
    counts={}
    for line in printed.splitlines():
        match=re.search(r'with (\d+) rows .*? - (/world/[^ ]+) - data columns: \[(.*)\]',line)
        if not match:continue
        n,entity,columns=match.groups()
        for field in ['Mesh3D:vertex_positions','Transform3D:translation','Pinhole:image_from_camera','EncodedImage:blob','Clear:is_recursive']:
            if field in columns:
                key=entity+' '+field;counts[key]=counts.get(key,0)+int(n)
    expected={'/world/camera Transform3D:translation':150,'/world/camera/image Pinhole:image_from_camera':150,'/world/camera/image EncodedImage:blob':150,'/world/hands/left Mesh3D:vertex_positions':150,'/world/hands/right Mesh3D:vertex_positions':150,'/world/object Transform3D:translation':148,'/world/object Clear:is_recursive':2}
    for key,n in expected.items():
        if counts.get(key)!=n:raise RuntimeError((key,counts.get(key),n))
    assert sha(p)==rec['source_npz_sha256']
    results.append({'label':label,'path':target,'sha256':rec['sha256'],'bytes':rec['bytes'],'rrd_verified':True,'decoded_entity_rows':counts,'seconds':time.monotonic()-t})
    print('DONE',json.dumps(results[-1]),flush=True)
(out/'validation.json').write_text(json.dumps({'exports':results,'rerun_version':'0.37.2','source_inputs_unchanged':True,'self_contained_rgb_and_geometry':True,'viewer_opened_interactively':False,'candidate_only':True},indent=2)+'\n')
print('ALL_FIVE_VERIFIED',flush=True)
