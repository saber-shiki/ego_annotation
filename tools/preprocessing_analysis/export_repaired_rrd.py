"""Interactive complete native R6 / repaired / GT world comparison, post-freeze only."""
from pathlib import Path
import sys,json
import numpy as np
HOME=Path('/mnt/user-home/kupingxin');sys.path.insert(0,str(HOME/'ego_annotation_experiments/milk_R6_vs_CAD_GT_20261014'))
from evaluate_comparison import load,read,sha,PRED,GT,CAD,mesh,apply
import rerun as rr
import rerun.blueprint as rb
HERE=HOME/'ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1'
FINAL=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/milk_shape_depth_hands_20261015_r1/stage3_hands_r2/coarse_scene.npz')

def main():
 assert (FINAL.parent/'completed.json').exists();b=load(PRED);p=load(FINAL);g=load(GT);cad=mesh(CAD);oi=int(np.flatnonzero(g['object_uids']=='125863066770940')[0]);D=g['T_world_from_camera'];G=g['T_world_from_object'][:,oi];A=D[0]@np.linalg.inv(b['T_world_camera'][0])
 hashes={str(x):sha(x) for x in [PRED,FINAL,GT,CAD]};rec=rr.RecordingStream('milk_preprocess_ordered_repair');rec.set_log_time_enabled(False);rec.set_log_tick_enabled(False);out=HERE/'final_review/baseline_repair_GT.rrd';rec.save(out)
 families=[('R6',b,[40,137,191]),('repaired',p,[12,167,185])]
 pts=[]
 for family,a,color in families:
  rec.log(f'/world/{family}/object/mesh',rr.Mesh3D(vertex_positions=a['vertices_canonical'].astype(np.float32),triangle_indices=a['faces'].astype(np.uint32),albedo_factor=color+[255],face_rendering=rr.components.MeshFaceRendering.DoubleSided),static=True)
  pos=apply(A,a['object_translation_world']);pts.append(pos)
 rec.log('/world/GT/object/mesh',rr.Mesh3D(vertex_positions=cad.vertices.astype(np.float32),triangle_indices=cad.faces.astype(np.uint32),albedo_factor=[234,136,34,255],face_rendering=rr.components.MeshFaceRendering.DoubleSided),static=True)
 rows=read(HERE/'stage3_evaluation/frame_metrics.json')
 for i in range(150):
  rec.set_time('frame',sequence=i);rec.set_time('time_s',duration=i/30)
  for family,a,color in families:
   O=np.eye(4);O[:3,:3]=a['object_rotation'][i];O[:3,3]=a['object_translation_world'][i];O=A@O;C=A@a['T_world_camera'][i]
   rec.log(f'/world/{family}/object',rr.Transform3D(translation=O[:3,3],mat3x3=O[:3,:3]));rec.log(f'/world/{family}/camera',rr.Transform3D(translation=C[:3,3],mat3x3=C[:3,:3]));rec.log(f'/world/{family}/camera/pinhole',rr.Pinhole(focal_length=a['intrinsics_source'][i,:2],principal_point=a['intrinsics_source'][i,2:],resolution=a['image_size_wh'][i],image_plane_distance=.04))
   for s,col in [('left',[44,107,183]),('right',[77,163,220])]:
    V=apply(A,a[s+'_vertices'][i]).astype(np.float32);J=apply(A,a[s+'_joints'][i]).astype(np.float32)
    rec.log(f'/world/{family}/{s}',rr.Mesh3D(vertex_positions=V,triangle_indices=a[s+'_faces'].astype(np.uint32),albedo_factor=col+[255],face_rendering=rr.components.MeshFaceRendering.DoubleSided));rec.log(f'/world/{family}/{s}/joints',rr.Points3D(J,colors=col,radii=.001))
  rec.log('/world/GT/object',rr.Transform3D(translation=G[i,:3,3],mat3x3=G[i,:3,:3]));rec.log('/world/GT/camera',rr.Transform3D(translation=D[i,:3,3],mat3x3=D[i,:3,:3]));rec.log('/world/GT/camera/pinhole',rr.Pinhole(focal_length=g['intrinsics_fx_fy_cx_cy'][:2],principal_point=g['intrinsics_fx_fy_cx_cy'][2:],resolution=g['image_size_wh'],image_plane_distance=.04))
  for s,col in [('left',[185,87,31]),('right',[232,173,63])]:
   si=int(np.flatnonzero(g['mano_sides']==s)[0]);rec.log(f'/world/GT/{s}',rr.Mesh3D(vertex_positions=g['mano_world_vertices_m'][i,si],triangle_indices=g['mano_'+s+'_faces'].astype(np.uint32),albedo_factor=col+[255],face_rendering=rr.components.MeshFaceRendering.DoubleSided))
  r=rows[i];rec.log('/notes',rr.TextDocument(f"# frame {i} / source {2370+i}\nR6 / ordered repair / recorded GT. Complete native mesh + FK MANO, metres.\n\nSame first-camera SE3 only, no scale or object fit.\n\nRepair hand world MPJPE19 L/R {r['left_MPJPE19_world_mm']:.2f}/{r['right_MPJPE19_world_mm']:.2f}mm. Object camera surface {r['object_surface_camera_mean_mm']:.2f}mm.\n\nNot contact or zero-penetration certification.",media_type='text/markdown'))
 center=np.median(G[:,:3,3],axis=0)+G[0,:3,1]*.07;eye=center+D[0,:3,:3]@np.array([.24,-.12,-.58]);up=D[0,:3,:3]@np.array([0,-1,0]);ec=rb.EyeControls3D(position=eye,look_target=center,eye_up=up)
 def view(family):return rb.Spatial3DView(origin='/world',name=family,contents=[f'/world/{family}/**'],eye_controls=ec,background=[246,248,251],line_grid=False,defaults=[rr.TransformAxes3D(axis_length=0)])
 bp=rb.Blueprint(rb.Vertical(rb.Horizontal(view('R6'),view('repaired'),view('GT')),rb.TextDocumentView(origin='/notes',name='Frame / coordinates / uncertainty'),row_shares=[.85,.15]),rb.TimePanel(timeline='frame',state='expanded',fps=30,play_state='paused'),rb.SelectionPanel(state='collapsed'),auto_views=False,auto_layout=False)
 rec.send_blueprint(bp,make_active=True,make_default=True);rec.flush();rec.disconnect()
 for path,h in hashes.items():assert sha(path)==h
 save=dict(path=str(out),sha256=sha(out),frames=150,object_transforms=450,camera_transforms=450,MANO_mesh_frames=900,source_sha256=hashes,prediction_used_only_after_freeze=True,geometry='all native original vertices/faces; hand FK surfaces',world_alignment='single first-camera SE3, no fitted scale or object canonical change')
 (out.with_suffix('.receipt.json')).write_text(json.dumps(save,indent=2)+'\n');print('RRD exported',out)
if __name__=='__main__':main()
