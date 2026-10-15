"""Decode complete comparison videos and retain all-frame inspection sheets."""
from pathlib import Path
import argparse,json,hashlib
import cv2
import numpy as np


def inspect(run,out):
    out.mkdir(parents=True,exist_ok=False)
    receipts={}
    for name in ['zero_best_last_overlay','zero_best_last_world']:
        path=run/'renders'/f'{name}.mp4'
        cap=cv2.VideoCapture(str(path));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None:raise RuntimeError('empty decoded frame')
            frames.append(frame)
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if len(frames)!=150:raise RuntimeError('missing comparison frames')
        short='overlay' if name.endswith('overlay') else 'world'
        for start in (0,50,100):
            tiles=[cv2.resize(x,(384,round(x.shape[0]*.5)),interpolation=cv2.INTER_AREA) for x in frames[start:start+50]]
            rows=[np.concatenate(tiles[j:j+5],axis=1) for j in range(0,50,5)]
            cv2.imwrite(str(out/f'{short}_{start:03d}_{start+49:03d}.jpg'),np.concatenate(rows,axis=0))
        for i in (0,30,60,90,120,143,148,149):cv2.imwrite(str(out/f'{short}_{i:03d}.jpg'),frames[i])
        receipts[name]={'frames':len(frames),'fps':fps,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'verification.json').write_text(json.dumps({'scope':'read_only_full_native_MANO_video_review','decoded_videos':receipts},indent=2)+'\n')
    print(json.dumps(receipts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
