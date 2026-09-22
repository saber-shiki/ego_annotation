"""QC views of actual source arrays. Not a policy renderer or annotation export."""
import cv2
import numpy as np


def panel(image, title, width=480):
    height=max(1,round(image.shape[0]*width/image.shape[1]))
    canvas=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
    canvas=cv2.copyMakeBorder(canvas,42,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(canvas,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return canvas


def mask_view(image, mask, color):
    canvas=image.copy()
    if mask is not None:
        take=cv2.resize(mask.astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        canvas[take]=(canvas[take]*.55+np.asarray(color)*.45).astype(np.uint8)
    return canvas


def triangles(canvas, uv, valid, faces, color, *, fill=False):
    faces=np.asarray(faces,np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and (faces.min()<0 or faces.max()>=len(uv))):
        raise ValueError('QC requires complete, valid triangle indices')
    finite=valid & np.isfinite(uv).all(axis=1) & (np.max(np.abs(uv),axis=1)<1e7)
    eligible=finite[faces].all(axis=1)
    polygons=np.rint(uv[faces[eligible]]).astype(np.int32)
    if len(polygons):
        if fill:
            # One convex triangle per call avoids even/odd cancellation of neighboring faces.
            overlay=canvas.copy()
            for triangle in polygons:cv2.fillConvexPoly(overlay,triangle,color)
            canvas[:]=cv2.addWeighted(canvas,.55,overlay,.45,0.)
        else:
            cv2.polylines(canvas,polygons,True,color,1,cv2.LINE_AA)
    return {'faces_total':len(faces),'faces_projectable':int(eligible.sum()),
            'faces_not_projectable':int((~eligible).sum()),'near_plane_crossing_faces_not_clipped':True}


def source_view(image, arrays, position, appearance_prepared=None, visual=None):
    size=np.asarray(arrays['source_visual_preprocessing']['fit_size_wh'])
    canvas=cv2.resize(image,tuple(size),interpolation=cv2.INTER_AREA)
    counts={}
    for side,color in [('left',(50,220,50)),('right',(30,150,255))]:
        source=arrays['source_visual_preprocessing']['hands'][side]
        if source['source_valid'][position]:
            counts[side]=triangles(canvas,source['vertices_uv_fit'][position],
                source['vertex_projection_valid'][position],arrays['hands'][side]['faces'],color)
            ids=np.unique(source['pad_vertex_ids'][position])
            ids=ids[(ids>=0)&source['vertex_projection_valid'][position,ids.clip(min=0)]]
            for uv in source['vertices_uv_fit'][position,ids]:
                if np.isfinite(uv).all() and np.max(np.abs(uv))<1e7:
                    cv2.circle(canvas,tuple(np.rint(uv).astype(int)),2,(255,0,255),-1)
        else:counts[side]={'source_valid':False}
    if appearance_prepared is not None and appearance_prepared.get('available'):
        contour=appearance_prepared['contour_uv']
        if len(contour):cv2.polylines(canvas,[np.rint(contour).astype(np.int32)],True,(255,255,0),2)
    if visual is not None and int(arrays['frame_idx'][position])==visual.anchor_frame_idx:
        camera=visual.vertices_metric_camera;K=arrays['source_visual_preprocessing']['intrinsics_fit'][position]
        valid=np.isfinite(camera).all(axis=1)&(camera[:,2]>1e-5)
        uv=np.zeros((len(camera),2));uv[valid]=camera[valid,:2]/camera[valid,2:]*K[:2]+K[2:]
        counts['complete_native_at_anchor']=triangles(canvas,uv,valid,visual.faces,(255,100,50),fill=True)
    return canvas,counts


def depth_view(depth):
    valid=np.isfinite(depth)&(depth>0)
    if not valid.any():return np.zeros((*depth.shape,3),np.uint8),None
    low,high=np.percentile(depth[valid],[2,98]);gray=np.zeros(depth.shape,np.uint8)
    gray[valid]=np.clip((depth[valid]-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8)
    image=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);image[~valid]=0
    return image,[float(low),float(high)]


def edge_view(source_image,target_image,edge,projected_target,max_points=32):
    left=panel(source_image,f'RGB source {edge.source_frame_idx}')
    right=panel(target_image,f'target {edge.target_frame_idx}: observed / projected')
    height=max(left.shape[0],right.shape[0])
    left=cv2.copyMakeBorder(left,0,height-left.shape[0],0,0,cv2.BORDER_CONSTANT)
    right=cv2.copyMakeBorder(right,0,height-right.shape[0],0,0,cv2.BORDER_CONSTANT)
    out=np.concatenate([left,right],axis=1);w=left.shape[1]
    def display(uv,size,image):
        shape=np.array([w,round(image.shape[0]*w/image.shape[1])]);return (uv+.5)*shape/size-.5+np.array([0,42])
    su=display(edge.source_uv,edge.source_size_wh,source_image)
    tu=display(edge.target_uv,edge.target_size_wh,target_image)+[w,0]
    pu=display(projected_target,edge.target_size_wh,target_image)+[w,0]
    take=np.linspace(0,len(su)-1,min(max_points,len(su)),dtype=int) if len(su) else []
    for i in take:
        a,b,c=[tuple(np.rint(x).astype(int)) for x in (su[i],tu[i],pu[i])]
        cv2.line(out,a,b,(140,100,100),1,cv2.LINE_AA)
        cv2.circle(out,a,3,(255,0,255),-1);cv2.circle(out,b,3,(0,255,0),1)
        cv2.drawMarker(out,c,(0,160,255),cv2.MARKER_CROSS,6,1)
    return out
