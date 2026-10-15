"""Summarize frozen ordered interventions; reference data only in this report."""
from pathlib import Path
import json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HOME=Path('/mnt/user-home/kupingxin');HERE=HOME/'ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1';OUT=HERE/'final_review';RUN=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/milk_shape_depth_hands_20261015_r1')
records=[('R6',HOME/'ego_annotation_experiments/milk_R6_vs_CAD_GT_20261014',None),('Shape',HERE/'stage1_r2_evaluation','stage1_shape_r2'),('Depth only',HERE/'stage2_r2_evaluation','stage2_depth_only_r2'),('Shape+depth',HERE/'combined_r2_evaluation','stage2_shape_depth_r2'),('MANO joint',HERE/'stage3_evaluation','stage3_hands_r2')]
rows=[];detail={};allframes={}
for label,path,stage in records:
 j=json.loads((path/'metrics.json').read_text());frames=json.loads((path/'frame_metrics.json').read_text());detail[label]=j;allframes[label]=frames;M=j['metrics'];dims=j['object_size']['pred_sorted_OBB_mm']
 item=dict(stage=label,run=str(RUN/stage) if stage else j['prediction_path'],short_mm=dims[0],long_mm=dims[1],height_mm=dims[2],object_surface_camera_mm=M['object_surface_camera_mean_mm']['mean'],object_surface_world_mm=M['object_surface_world_mean_mm']['mean'],object_depth_bias_mean_mm=j['object_centroid_camera_bias_mean_xyz_mm'][2],object_depth_bias_tail30_mm=float(np.mean([r['object_centroid_camera_bias_xyz_mm'][2] for r in frames[-30:]])),object_depth_bias_tail10_mm=float(np.mean([r['object_centroid_camera_bias_xyz_mm'][2] for r in frames[-10:]])),object_depth_bias_f147_mm=frames[147]['object_centroid_camera_bias_xyz_mm'][2],object_motion_camera_rotation_deg=M['object_motion_camera_rotation_deg']['mean'])
 if stage:
  d=json.loads((RUN/stage/'source_diagnostics.json').read_text());item['source_visible_mask_IoU']=float(np.mean([r['mask']['visible']['iou'] for r in d['after']]))
 else:item['source_visible_mask_IoU']=float(np.mean([r['mask']['visible']['iou'] for r in json.loads((RUN/'stage1_shape_r2/source_diagnostics.json').read_text())['before']]))
 for s in ['left','right']:
  item[s+'_MPJPE19_world_mm']=M[s+'_MPJPE19_world_mm']['mean'];item[s+'_MPJPE19_camera_mm']=M[s+'_MPJPE19_camera_mm']['mean'];item[s+'_accel_error_m_s2']=M[s+'_acceleration_error_m_s2']['mean'];item[s+'_actual_accel_m_s2']=M[s+'_pred_acceleration_m_s2']['mean'];item[s+'_projection_error_px']=M[s+'_joint_projection_mean_px']['mean'];item[s+'_wrist_depth_tail30_mm']=float(np.mean([r[s+'_wrist_camera_bias_xyz_mm'][2] for r in frames[-30:]]))
 rows.append(item)
(OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
b=rows[0];f=rows[-1]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(13,8));colors=['#798493','#41a382','#9279bc','#1ba8ba','#0d728e']
for (label,_,_),c in zip(records,colors):
 r=allframes[label];t=[x['time_s'] for x in r];axes[0,0].plot(t,[x['object_centroid_camera_bias_xyz_mm'][2] for x in r],label=label,color=c,lw=1.2);axes[0,1].plot(t,[x['object_surface_camera_mean_mm'] for x in r],label=label,color=c,lw=1.2)
for s,c in [('left','#267ab0'),('right','#c28c43')]:
 for label,style in [('R6','--'),('MANO joint','-')]:
  r=allframes[label];axes[1,0].plot([x['time_s'] for x in r],[x[s+'_MPJPE19_world_mm'] for x in r],color=c,ls=style,label=label+' '+s)
labels=['R6','Shape','Depth only','Shape+depth','MANO joint'];x=np.arange(5);axes[1,1].bar(x-.17,[r['left_accel_error_m_s2'] for r in rows],.34,label='left');axes[1,1].bar(x+.17,[r['right_accel_error_m_s2'] for r in rows],.34,label='right');axes[1,1].set_xticks(x,labels,rotation=15);axes[1,1].set_ylabel('mean acceleration error m/s²');axes[1,1].set_title('Actual MANO motion, not only correction smoothness')
for ax,title,y in [(axes[0,0],'Object depth bias in matched camera','mm, + = too far'),(axes[0,1],'Object surface mean in matched camera','mm'),(axes[1,0],'Hand world MPJPE19 | one first-camera SE3','mm')]:ax.set_title(title);ax.set_xlabel('time (s)');ax.set_ylabel(y);ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[1,1].legend();fig.suptitle('Milk ordered preprocessing repairs | independent post-freeze evaluation',fontsize=14);fig.tight_layout();fig.savefig(OUT/'summary_curves.png',dpi=150);plt.close(fig)
objecttable='\n'.join(f"|{r['stage']}|{r['short_mm']:.2f}×{r['long_mm']:.2f}×{r['height_mm']:.2f}|{r['object_surface_camera_mm']:.2f}|{r['object_depth_bias_tail30_mm']:.2f}|{r['object_motion_camera_rotation_deg']:.2f}|{r['source_visible_mask_IoU']:.4f}|" for r in rows)
report=f'''# Milk 预处理顺序修复：实际结果与未解问题

## 结论

**已在新worktree实施、完整运行和渲染三项机制。物体横向比例与后段错误退远得到改善，但尺寸仍未完全准确；双手明显更平滑，绝对手部深度只得到有限改善。不是目标全解决，也没有替换默认R6或继续训练RL。**

- 新worktree：`/mnt/user-home/kupingxin/ego_annotation_preprocess_shape_depth_hands`
- 分支：`feature/milk-preprocess-shape-depth-hands-20261015`，基于6c4fbab；继承原R6未提交补丁，未commit/merge/push。原工作树397份记录文件hash不变。
- 冻结快照：`/mnt/user-home/kupingxin/ego_annotation_runtime/milk_preprocess_shape_depth_hands_20261015_*`，不同修正分别冻结，未运行中改代码。
- 预测根：`{RUN}`，原v2 source和R6只读。算力host=`dexrobot-Standard-PC-Q35-ICH9-2009`；CPU单线程、禁CUDA，无新感知推理/训练/安装。
- 视频：P0014_84ea2dcc milk，原2370–2519，150帧/30fps/5秒。
- GT/CAD只在输出冻结后用于评价/对比，不作为尺寸、深度曲线、权重或手部拟合目标。主世界坐标只使用一次首帧相机SE3，scale1，无逐帧或物体拟合。

## 1. 物体各阶段完整结果

尺寸为排序OBB的短横向×长横向×高度；不是语义宽深。表面距离每帧4096点/方向，对真实完整三角面双向距离，不是质心或盒子。Z偏差为预测/GT各自同帧相机中的表面积几何中心差；+表示过远。

|阶段|尺寸mm|相机表面均值mm|末1秒平均Z偏差mm|旋转运动差均值°|源可见mask IoU|
|---|---|---:|---:|---:|---:|
{objecttable}
|CAD GT|72.08×72.56×192.08|—|—|—|—|

旋转运动差去掉固定首帧canonical偏置；只是运动指标，未用于世界图。

- 原长横向+20.94%改善为**+10.08%**，短横向−6.95%改善为**−1.14%**。高度由+0.83%变成**−1.69%**：高度略变小，这个残差不能隐瞒或称全尺寸已准确。
- 最终物体相机表面均值**13.91→7.30mm（下降约47.5%）**；世界表面均值16.09→9.22mm。
- 末1秒平均错误退远**36.13→12.47mm（减少约65.5%）**，剩余仍非零。**最后10帧仍为46.74→23.46mm，f147仍有29.75mm过远误差**；不能用末1秒平均掩盖最后几帧的残余后退。绝对首帧米制gauge错误未被相对校准解决。
- 相机/K/timeline保持逐数组不变，因此深度收益不是把世界视图重新配准造成的。

## 2. 每一步真正改变了什么

### 形状 r2

完整239430顶点/478880面，沿源网格主轴的共享正定变换；改变的是实际所有顶点，不是primitive replacement。多关键帧的真实轮廓、首表面深度、可见三角面法向和弱RGB棱线证据；单一形状全视频共享。每步重新查询固定域整张轮廓，防止只在旧对应点上下降。未写入CAD尺寸，未给每帧单独尺度。

尺寸比例部分改善，仍有10%的长横向误差。单目深度和棱线观测有噪声，主轴共享形变不能恢复任意隐藏拓扑，也没有保证全局正确。

### 相对深度校准 / ordered组合

176条源RGB材料对应边，以点对弦长刚性约束估计7参数时间曲线，首帧scale严格1，无每帧additive bias。曲线范围0.9317–1.0，末1秒平均约0.939；144/150帧有至少2条对应边支持，缺/弱支持仍记录。

raw UniDepth未修改，校准后的云独立使用。depth-only冻结R6旋转/形状；组合保持第一步共享形状和旋转，只将相对尺度用于全时序平移。避免把深度校准重新吸收到物体绝对尺寸，或在错误模板下转动物体来掩盖深度。

### MANO联合修正

先精确replay源MANO，最大逐点偏差≤8.94e−8m；随后每侧一份β，完整root orientation、translation、45D articulation及object6D参与可微优化。真实RGB-LK对应、弱源投影prior、实际world加速度、图像已声明digit的保守contact约束。不是移动2D骨架，也不是仅平滑RL修正量。

左右5206/4646条像素观测，whole-tracklet holdout1326/1143；输出由完整FK驱动。源没有独立2D关节检测器，RGB track的三角面绑定依赖初始MANO，接触支持60mm是保守配置prior，并非确定真接触。没有用同一份不可靠UniDepth把双手一起拉到物体上。

## 3. 手部：时间改善明显，绝对深度仍有限

|指标|R6左|最终左|R6右|最终右|
|---|---:|---:|---:|---:|
|world MPJPE19 /mm|{b['left_MPJPE19_world_mm']:.2f}|{f['left_MPJPE19_world_mm']:.2f}|{b['right_MPJPE19_world_mm']:.2f}|{f['right_MPJPE19_world_mm']:.2f}|
|camera MPJPE19 /mm|{b['left_MPJPE19_camera_mm']:.2f}|{f['left_MPJPE19_camera_mm']:.2f}|{b['right_MPJPE19_camera_mm']:.2f}|{f['right_MPJPE19_camera_mm']:.2f}|
|实际关节加速度误差 /m/s²|{b['left_accel_error_m_s2']:.3f}|{f['left_accel_error_m_s2']:.3f}|{b['right_accel_error_m_s2']:.3f}|{f['right_accel_error_m_s2']:.3f}|
|末1秒腕点Z偏差 /mm|{b['left_wrist_depth_tail30_mm']:.2f}|{f['left_wrist_depth_tail30_mm']:.2f}|{b['right_wrist_depth_tail30_mm']:.2f}|{f['right_wrist_depth_tail30_mm']:.2f}|
|GT关节投影均值 /px|{b['left_projection_error_px']:.2f}|{f['left_projection_error_px']:.2f}|{b['right_projection_error_px']:.2f}|{f['right_projection_error_px']:.2f}|

- 实际加速度误差左约**下降83.7%**，右约**下降87.7%**；恒定β已满足。同视频真实运动未整体取消。
- MPJPE绝对误差只是小幅下降；左腕末段稍改善，右腕末段略更近，并未修复右手绝对深度。
- **GT投影误差略变差**（左+0.43px、右+0.15px）。源RGB track一致性/平滑改善不能当作GT准确度全面提升；不推广MANO联合结果为新默认手部。
- 可在入口分别比较“Shape+depth”（原双手）与“MANO joint”（联合双手）。当前较有依据的收益来自物体相对深度，不是已经解决全部手物物理状态。

## 4. 实际失败/退化记录

1. `stage1_shape` r1：横向尺寸改善但高度压缩、IoU下降；保留全视频和评价，不采用。固定对应点下降不保证真实轮廓改善。r2加入fresh轮廓检查及源面法向/共享shape先验，降低该退化，但高度仍有残差。
2. `stage2_depth_only` r1：深度改善，旋转差6.59→9.89°；这是自由旋转补偿污染depth-only干预。保留；r2冻结旋转，旋转差保持6.59°而深度收益仍存在。
3. `stage2_shape_depth` free联合r1：表面可降至6.31mm，却把高度压到179.76mm、旋转误差8.42°；不因表面分数更低就选它。ordered组合冻结已估计共享形状/旋转，不重新拟合绝对尺寸。
4. `stage3_hands` r1：数值优化完成，但导出变量shadow导致TypeError，未产生完整预测；失败JSON/log保留。仅修正命名，在新snapshot/新root以同参数运行`stage3_hands_r2`，完整导出/渲染。

这些是同视频有限机制研究，非泛化/收敛证据。没有commit/merge/push，不把失败替换为成功，也没有增加RL训练步数。

## 5. 实际可视化

- [入口index.html](index.html)
- [R6 / 最终联合 / CAD+MANO GT，完整世界视频](baseline_final_GT_world.mp4)
- [五个阶段+GT，完整世界视频](all_stages_world.mp4)
- [五个阶段，完整RGB overlay](all_stages_overlay.mp4)
- [可旋转、逐帧完整网格RRD](baseline_repair_GT.rrd)
- [误差曲线](summary_curves.png)、[逐阶段CSV](summary.csv)、[摘要JSON](summary.json)
- [f88手部](baseline_final_GT_f088.jpg)、[f147后段](baseline_final_GT_f147.jpg)

世界图统一为GT world，只做首帧一次SE3，保留scale1，没有按物体逐帧对齐。每阶段完整RGB/world视频150帧；全阶段对比两视频150帧；主视频150帧。完整原生三角面、双手778顶点；RRD450物体poses/450相机poses/900MANO帧均读回核对。独立MANO参数FK和源array/原树/快照hash验证保证数据连接，不认证真实接触或零穿透。

## 6. 保留的边界和下一步

- 物体横截面仍约+10%长边误差，高度约−1.7%；需更可靠的可见面/内部棱线对应和绝对米制锚点，不能把CAD尺寸硬写入管线。
- 相对深度降低后段漂移，但锚点自身误差保留；短运动单目不能保证绝对尺度唯一。
- 手部缺独立绝对图像关节/深度支持，当前主要修复时间噪声和恒定形状；下一步应补独立手部观测与visibility，而非提高contact权重把手强拉到有误差的物体上。
- 源generated mesh signed/contact权威没有升级，未做零穿透认证，未把新manifest喂入旧actor或删除guard。保持R6默认，Shape+depth和MANO joint均为可审查候选，等待用户判断。

代码机制：`preprocessing/shared_repair.py`、`relative_repair.py`、`mano_repair.py`、`sequential_repair.py`、`repair_render.py`。冻结实际命令/日志在本报告上一级。核心回归150项（146通过/4skip），另24项shape/mask测试和3项relative测试通过；FK/derivative smoke通过。测试不代替物理效果。
'''
(OUT/'REPORT_ZH.md').write_text(report);(HOME/'ego_annotation_preprocess_shape_depth_hands/docs/SHAPE_DEPTH_HAND_REPAIR_RESULTS_20261015.md').write_text(report)
tr=''.join(f"<tr><td>{r['stage']}</td><td>{r['short_mm']:.1f} × {r['long_mm']:.1f} × {r['height_mm']:.1f}</td><td>{r['object_surface_camera_mm']:.2f}</td><td>{r['object_depth_bias_tail30_mm']:.2f}</td><td>{r['left_MPJPE19_world_mm']:.2f} / {r['right_MPJPE19_world_mm']:.2f}</td></tr>" for r in rows)
stage_links=[]
for label,path,stage in records:
 if not stage:continue
 for filename in ['before_after_overlay.mp4','before_after_world.mp4']:
  dst=OUT/(stage+'_'+filename)
  if not dst.exists():dst.symlink_to(RUN/stage/'renders'/filename)
 stage_links.append(f'<h3>{label}，对比冻结R6</h3><p><a href="{stage}_before_after_overlay.mp4">完整RGB overlay</a> · <a href="{stage}_before_after_world.mp4">完整world</a></p>')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Milk顺序预处理修复</title><style>body{{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 20px;background:#f5f8fa;color:#20323d}}h1,h2{{color:#13667b}}video,img{{width:100%;border:1px solid #d8e1e8;border-radius:7px;background:white}}table{{border-collapse:collapse;background:white}}td,th{{border:1px solid #ddd;padding:10px}}.note{{padding:17px;background:#e6f4f4;border-left:5px solid #159bad}}a{{color:#087b91}}code{{word-break:break-all}}</style><h1>Milk：新worktree上的顺序预处理修复</h1><p>150帧 · 30fps · 5秒 · 2026-10-15</p><div class="note"><b>物体深度改善明显，尺寸仅部分改善；手部主要改善抖动，绝对深度未解决。</b>不替换默认R6，不训练RL，不认证接触或零穿透。</div><p>World仅一次首帧相机SE3，无scale/object/per-frame fit。蓝青为预测、橙黄为GT；保持全部原生三角面和完整MANO。</p><table><tr><th>阶段</th><th>OBB尺寸mm</th><th>物体camera表面mm</th><th>末1秒物体Z偏差mm</th><th>World手部MPJPE左/右mm</th></tr>{tr}<tr><td>CAD GT</td><td>72.1×72.6×192.1</td><td>—</td><td>—</td><td>—</td></tr></table><p><a href="REPORT_ZH.md">完整报告：含退化和失败</a> · <a href="summary.csv">CSV</a> · <a href="summary.json">JSON</a> · <a href="baseline_repair_GT.rrd" download>可旋转逐帧3D RRD</a> · <a href="verification.json">实际几何/参数验证</a></p><h2>完整世界：R6 / 最终联合 / GT</h2><video controls loop muted preload="metadata" src="baseline_final_GT_world.mp4"></video><h2>逐阶段完整世界（五阶段+GT）</h2><video controls loop muted preload="metadata" src="all_stages_world.mp4"></video><h2>逐阶段完整RGB overlay（五阶段）</h2><video controls loop muted preload="metadata" src="all_stages_overlay.mp4"></video><h2>独立阶段视频</h2>{''.join(stage_links)}<h2>误差曲线</h2><img src="summary_curves.png"><h2>f147 后段</h2><img src="baseline_final_GT_f147.jpg"><h2>f88 手部异常</h2><img src="baseline_final_GT_f088.jpg"><p>末段物体Z偏差36.13→12.47mm；最后10帧仍为46.74→23.46mm，f147仍过远29.75mm。尺寸71.26×79.87×188.83mm，长横边仍+10%，高度−1.7%。手实际加速度误差显著下降，手绝对MPJPE只小幅下降且GT投影稍差；不能宣称全解决。</p><p>新worktree：<code>/mnt/user-home/kupingxin/ego_annotation_preprocess_shape_depth_hands</code>，未commit/merge/push，原树及冻结预测不变。</p></html>'''
(OUT/'index.html').write_text(page)
print(json.dumps(rows,indent=2));print('report/curves/index written')
