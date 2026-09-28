"""Read-only scale visualization: same origin/orientation and global mm/pixel ruler.
Uses actual source/output triangles, not image generation or independently normalized views.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,sys,gc
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular')
from preprocessing.geometry import TriangleMesh
ROOT=Path(__file__).resolve().parent
RUN=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260927_r2')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
cv2.setNumThreads(1)

def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)
def read(p):return json.loads(Path(p).read_text())

def render(vertices,faces,cx,cy,z0,ppm,width,height,color):
    mesh=TriangleMesh(vertices,faces)
    yy,xx=np.mgrid[:height,:width]
    origins=np.stack(((xx-(width-1)/2)/ppm+cx,(yy-(height-1)/2)/ppm+cy,np.full_like(xx,z0,dtype=float)),axis=-1)
    direction=np.zeros_like(origins);direction[:,:,2]=1.
    rays=np.concatenate((origins,direction),axis=-1).astype('float32')
    hit=mesh.scene.cast_rays(mesh.o3d.core.Tensor(rays),nthreads=1)
    depth=hit['t_hit'].numpy();mask=np.isfinite(depth)&(depth>0)
    normals=hit['primitive_normals'].numpy();light=np.array([-.35,-.45,-1.]);light/=np.linalg.norm(light)
    shade=.38+.62*np.abs(normals@light)
    canvas=np.full((height,width,3),248,np.uint8)
    canvas[mask]=np.clip(np.array(color)[None]*shade[mask,None],0,255).astype('uint8')
    del mesh;gc.collect()
    return canvas,mask


def main():
    items=read(RUN/'inputs.json');records=[];max_xy=np.zeros(2)
    for item in items:
        out=Path(item['output']);m=read(out/'coarse_scene.json')
        with np.load(out/'coarse_scene.npz',allow_pickle=False) as z:
            before=np.array(z['source_vertices_canonical']);after=np.array(z['vertices_canonical']);s=float(z['uniform_object_scale'])
            np.testing.assert_allclose(after,before*s,rtol=0,atol=1e-10)
            lo=np.minimum(before.min(0),after.min(0));hi=np.maximum(before.max(0),after.max(0));max_xy=np.maximum(max_xy,(hi-lo)[:2])
            records.append({'case':item['case'],'directory':str(out),'label':m['object_id'].replace('_',' ').title(),'scale':s,'lo':lo.tolist(),'hi':hi.tolist(),
                'before_span_mm':(np.ptp(before,axis=0)*1000).tolist(),'after_span_mm':(np.ptp(after,axis=0)*1000).tolist()})
    width,height=344,300;gap=18;pad=22;row_height=386;header=100;footer=56
    ppm=float(min((width-48)/max_xy[0],(height-30)/max_xy[1]))
    canvas=Image.new('RGB',(2*pad+3*width+2*gap,header+row_height*len(items)+footer),'white');draw=ImageDraw.Draw(canvas)
    draw.text((pad,14),'SAM3D mesh: scale-only comparison',font=font(27,True),fill=(25,35,48))
    draw.text((pad,52),'Same origin and view within each row; one metric ruler across ALL panels.',font=font(17),fill=(60,70,82))
    draw.text((pad,77),'Translations and pose updates are excluded. Geometry is rendered from actual triangles.',font=font(16),fill=(60,70,82))
    blue=(45,124,194);orange=(225,132,42)
    for index,rec in enumerate(records):
        out=Path(rec['directory']);lo=np.asarray(rec['lo']);hi=np.asarray(rec['hi']);center=(lo+hi)/2;z0=lo[2]-.1
        with np.load(out/'coarse_scene.npz',allow_pickle=False) as z:
            a,am=render(z['source_vertices_canonical'],z['faces'],center[0],center[1],z0,ppm,width,height,blue)
            b,bm=render(z['vertices_canonical'],z['faces'],center[0],center[1],z0,ppm,width,height,orange)
        overlay=np.full_like(a,248);overlay[bm]=(np.array(orange)*.40+248*.60).astype('uint8')
        for mask,color in [(am,blue),(bm,orange)]:
            contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,color,2)
        row_y=header+index*row_height
        change=(rec['scale']-1)*100
        draw.text((pad,row_y+3),f"{rec['label']}   scale x{rec['scale']:.4f}   ({change:+.1f}%)",font=font(21,True),fill=(28,36,46))
        titles=['Before: source mesh','After: aligned mesh','Overlay: blue before / orange after']
        for j,img in enumerate([a,b,overlay]):
            x=pad+j*(width+gap);y=row_y+55
            draw.text((x,row_y+31),titles[j],font=font(15),fill=(45,55,65));canvas.paste(Image.fromarray(img),(x,y))
            # All panels use exactly the same metric view and the same source pivot.
            ox=x+(width-1)/2-center[0]*ppm;oy=y+(height-1)/2-center[1]*ppm
            draw.line((ox-4,oy,ox+4,oy),fill=(85,85,85),width=1);draw.line((ox,oy-4,ox,oy+4),fill=(85,85,85),width=1)
            bar=50/1000*ppm;bx=x+12;by=y+height-19
            draw.rectangle((bx-6,by-25,bx+bar+14,by+10),fill=(255,255,255))
            draw.line((bx,by,bx+bar,by),fill=(25,25,25),width=3)
            draw.line((bx,by-4,bx,by+4),fill=(25,25,25),width=2);draw.line((bx+bar,by-4,bx+bar,by+4),fill=(25,25,25),width=2)
            draw.text((bx,by-24),'50 mm',font=font(14),fill=(25,25,25))
        text=lambda a:' / '.join(f'{x:.1f}' for x in a)
        draw.text((pad,row_y+height+60),f"Fixed-axis XYZ spans (mm): {text(rec['before_span_mm'])}  ->  {text(rec['after_span_mm'])}",font=font(15),fill=(65,70,78))
    y=header+row_height*len(items)+8
    draw.text((pad,y),'Units follow the prediction-side metric reference (UniDepth); these are not CAD/GT dimensions.',font=font(15),fill=(65,70,78))
    draw.text((pad,y+23),'Original source-appearance axes are preserved. XYZ spans depend on that fixed orientation.',font=font(15),fill=(65,70,78))
    png=ROOT/'sam3d_scale_only_comparison.png';jpg=ROOT/'sam3d_scale_only_comparison.jpg';canvas.save(png);canvas.save(jpg,quality=95)
    (ROOT/'scale_only_visualization.json').write_text(json.dumps({'method':'actual_triangle_orthographic_raycast','pixels_per_meter':ppm,'same_global_metric_ruler':True,'pose_and_translation_excluded':True,'source_files_modified':False,'cases':records,'png':str(png),'jpg':str(jpg)},indent=2)+'\n')
    print(png)

if __name__=='__main__':main()
