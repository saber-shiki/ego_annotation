"""Actual triangle raycast QC and full-timeline plots; no geometry fitting."""
from evaluate import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def raycast(s,T_camera_from_mesh,K,width,source_wh):
    # Resize intrinsics with integer pixel centers; ray parameter is camera-Z.
    scale=width/float(source_wh[0]);height=round(source_wh[1]*scale)
    k=np.array(K,float);k[:2]*=scale;k[2:]=(k[2:]+.5)*scale-.5
    y,x=np.mgrid[:height,:width];d=np.stack(((x-k[2])/k[0],(y-k[3])/k[1],np.ones_like(x)),axis=-1)
    inv=np.linalg.inv(T_camera_from_mesh);directions=d@inv[:3,:3].T
    origins=np.broadcast_to(inv[:3,3],directions.shape)
    rays=np.concatenate((origins,directions),axis=-1).astype(np.float32)
    z=s.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
    return np.isfinite(z)&(z>.001),z

def overlay(rgb,meshmask,cadmask,owned,title,caption):
    im=cv2.resize(rgb,meshmask.shape[::-1],interpolation=cv2.INTER_AREA)
    im[meshmask]=(im[meshmask]*.6+np.array([255,120,30])*.4).astype(np.uint8)
    for mask,col in ((owned,(0,225,245)),(cadmask,(40,245,50))):
        mask=cv2.resize(mask.astype('uint8'),meshmask.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im,contours,-1,col,1)
    top=np.zeros((66,im.shape[1],3),np.uint8)
    for text,yy,fs in [(title,18,.46),(caption,37,.36),('Blue: native  Green: CAD GT  Yellow: source mask',56,.32)]:cv2.putText(top,text,(6,yy),cv2.FONT_HERSHEY_SIMPLEX,fs,(245,245,245),1,cv2.LINE_AA)
    return np.vstack((top,im))

def iou(a,b):return float((a&b).sum()/max(1,(a|b).sum()))

if __name__=='__main__':
    cases=[r['case'] for r in read(ROOT/'integrity_summary.json')];sheet=[];allrender=[]
    fig,axes=plt.subplots(5,3,figsize=(15,17),squeeze=False)
    for index,case in enumerate(cases):
        c=load_case(case);rep=read(c['out']/'evaluation.json');nscene=scene(c['nv'],c['nf']);gscene=scene(c['gv'],c['gf']);panels=[]
        short=c['target']['object_name'];a=c['a'];metrics=[]
        for i,name in [(a,'anchor'),(119,'source'),(119,'seed')]:
            P=c['source' if name=='anchor' else name];H=np.linalg.inv(c['C'][i])@P[i]@np.linalg.inv(P[a])@c['C'][a]
            go=np.linalg.inv(c['D'][i])@c['G'][i]
            rgbpath=c['src']/Path(c['frames'][i]['raw_frame_path']).relative_to(c['original']);rgb=cv2.imread(str(rgbpath))
            if rgb is None:raise FileNotFoundError(rgbpath)
            ownedpath=c['src']/Path(c['objects'][i]['mask_path']).relative_to(c['original']);owned=cv2.imread(str(ownedpath),0)
            if owned is None:raise FileNotFoundError(ownedpath)
            owned=owned>0
            nm,nz=raycast(nscene,H,c['K'][i],360,[c['frames'][i]['source_width'],c['frames'][i]['source_height']]);gm,gz=raycast(gscene,go,c['K'][i],360,[c['frames'][i]['source_width'],c['frames'][i]['source_height']])
            owned_small=cv2.resize(owned.astype('uint8'),nm.shape[::-1],interpolation=cv2.INTER_NEAREST_EXACT)>0
            score=iou(nm,gm);caption=f'IoU(full CAD)={score:.3f}; object-only rendering'
            panels.append(overlay(rgb,nm,gm,owned,f'{short} f{i} {name}',caption));metrics.append({'frame':i,'mode':name,'native_vs_CAD_full_silhouette_IoU':score,'native_vs_owned_mask_IoU':iou(nm,owned_small),'CAD_vs_owned_mask_IoU':iou(gm,owned_small)})
        im=np.concatenate(panels,axis=1);cv2.imwrite(str(c['out']/'triangle_review.jpg'),im);sheet.append(im)
        (c['out']/'review_render_metrics.json').write_text(json.dumps({'case':case,'rows':metrics,'limitations':'360px full triangle first-hit raycast; object-only, no hand occlusion. Source mode reconstructed from frozen source pose and native appearance mesh, seed mode verified S3 preprocessed binding.'},indent=2)+'\n');allrender.append({'case':case,'rows':metrics})
        rows=rep['frame_rows']
        for name,col in [('source','tab:blue'),('seed','tab:orange')]:
            for j,key in enumerate(['aligned_camera_rotation_deg','aligned_camera_translation_mm']):
                yy=[r.get(name+'_'+key,np.nan) if r['GT_quality_eligible'] else np.nan for r in rows]
                axes[index,j].plot([r['frame'] for r in rows],yy,label=name,color=col,lw=1.5)
        axes[index,2].plot([r['frame'] for r in rows],[r.get('P09_depth_bias_median_mm',np.nan) for r in rows],color='tab:red',lw=1.3)
        for j,title in enumerate(['Anchor-aligned rotation / deg','Anchor-aligned translation / mm','P09 minus CAD depth / mm']):
            axes[index,j].set_title(f'{short}: {title}',fontsize=10);axes[index,j].set_xlabel('frame');axes[index,j].grid(alpha=.2)
        axes[index,0].legend(fontsize=8);axes[index,2].axhline(0,color='black',lw=.8)
        del c,nscene,gscene;gc.collect()
    cv2.imwrite(str(ROOT/'five_case_triangle_review.jpg'),np.concatenate(sheet,axis=0))
    fig.suptitle('Frozen S3 source vs RGB seed: motion tracking and retained-depth bias\nMotion scores use one GT anchor only; not absolute pose accuracy. Gaps remain gaps.',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(ROOT/'five_case_750_frame_diagnostics.png',dpi=130);plt.close(fig)
    (ROOT/'render_summary.json').write_text(json.dumps(allrender,indent=2)+'\n')
