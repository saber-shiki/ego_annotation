from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent/'monocular_visible_surface_20260928_jump_r2'
case=root/'milk_observation_v2_20260921_r2'
r=json.loads((case/'comparison.json').read_text());frames=r['frames'];x=[f['frame_idx'] for f in frames]
for name,lo,hi in [('full_timeline',0,149),('end_segment',135,149)]:
 fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
 for key,label,color in [('old','old contour-only output','#c44238'),('repaired','isolated evidence repair + gap-depth bridge','#007d9c')]:
  axes[0].plot(x,[f[key]['world_step_mm'] for f in frames],label=label,color=color,lw=1.6)
  axes[1].plot(x,[f[key]['rotation_step_deg'] for f in frames],color=color,lw=1.6)
  axes[2].plot(x,[f[key]['camera_translation_m'][2]*1000 for f in frames],color=color,lw=1.6)
 for ax in axes:
  ax.grid(alpha=.25);ax.set_xlim(lo,hi)
  ax.axvspan(142.7,143.3,color='#dddddd',alpha=.65)
 axes[0].set_ylabel('world position step (mm)');axes[1].set_ylabel('rotation step (deg)');axes[2].set_ylabel('object-origin camera Z (mm)')
 axes[0].legend(loc='upper left');axes[-1].set_xlabel('zero-based local frame index')
 fig.suptitle('Milk: actual delivered trajectory (not a post-hoc smoothed video)\nGray band: no qualified visible object surface at frame 143; raw depth was not modified')
 fig.tight_layout();fig.savefig(case/f'{name}.png',dpi=150);plt.close(fig)
print(case)
