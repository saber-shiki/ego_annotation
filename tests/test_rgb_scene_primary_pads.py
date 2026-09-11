"""Primary pad activation must couple to unsigned full-triangle fitting gradients."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v19_demo_timeline import primary_contact_pads
from fit_v20_visual_contact_demo import (visual_proximity_weights, primary_similarity_reachable,
    unsigned_pad_loss, HAND_SCALE_BOUNDS, HAND_TRANSLATION_BOUND_M)
from v20_demo_geometry import make_scene, closest, project


def payload(state='likely_contact', digits=None):
    row={'target_entity_id':'object:item','hand_side':'left','frame_start':10,'frame_end':80,
         'contact_state':state}
    if digits is not None:
        row['primary_contact_pads']={'digits':digits,'rationale':'Inspected frames show index pad support; occlusion uncertain.'}
    return {'interaction_judgments':[row]}


class PrimaryPads(unittest.TestCase):
    def test_interval_anatomy_and_invalid_declarations(self):
        ids=[10,37,80,123]
        selected=primary_contact_pads(payload(digits=['index','ring']),ids,'item')
        self.assertEqual(np.flatnonzero(selected[0,0]).tolist(),[1,3])
        self.assertFalse(selected[:,1].any());self.assertFalse(selected[-1].any())
        self.assertFalse(primary_contact_pads(None,ids,'item').any())
        for digits in (['index','index'],['wrist'],[1],'thumb'):
            with self.assertRaises(ValueError):primary_contact_pads(payload(digits=digits),ids,'item')
        bad=payload(digits=['index']);bad['interaction_judgments'][0]['primary_contact_pads']['rationale']=' '
        with self.assertRaises(ValueError):primary_contact_pads(bad,ids,'item')

    def test_noncontact_possible_unresolved_missing_visibility_do_not_expand(self):
        for state in ('no_contact','unresolved','possible_contact'):
            selected=primary_contact_pads(payload(state,['index']),[10],'item')
            self.assertFalse(selected.any())
            w=visual_proximity_weights([True]*5,[0.]*5,[.03]*5,state=state,primary=[True]*5,reachable=[True]*5)
            self.assertFalse(w.any())
        p=payload(digits=['index'])
        for state in ('out_of_frame','occluded','unresolved'):
            p['frames']=[{'frame_idx':10,'visibility':{'left':{'state':state},'object':{'state':'visible'}}}]
            self.assertFalse(primary_contact_pads(p,[10],'item').any())
        for designation in (None, [], ['index']):
            selected=primary_contact_pads(payload(digits=designation),[10],'item')[0,0]
            w=visual_proximity_weights([False]*5,[0.]*5,[.03]*5,primary=selected,reachable=[True]*5)
            self.assertFalse(w.any())
        self.assertFalse(visual_proximity_weights([True]*5,[0.]*5,[.03]*5,primary=[True]*5).any())
        self.assertFalse(visual_proximity_weights([True]*5,[0.]*5,[.03]*5,reachable=[True]*5).any())

    def test_primary_20_40mm_actual_triangle_gradient_and_finite_bound(self):
        scene=make_scene(np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]]),np.array([[0,1,2]]))
        for gap in (.020,.030,.040):
            points=np.tile([0.,0.,1.+gap],(5,1))
            target,_=closest(scene,points)
            reach=primary_similarity_reachable(points,target)
            self.assertTrue(reach.all())
            selected=primary_contact_pads(payload(digits=['index']),[10],'item')[0,0]
            weights=visual_proximity_weights([True]*5,[0.]*5,np.linalg.norm(points-target,axis=1),primary=selected,reachable=reach)
            self.assertEqual(np.flatnonzero(weights).tolist(),[1])
            # Same camera-origin scale, 12-sample pad and top-3 unsigned loss
            # as production, to a point inside a complete triangle (not vertices).
            logscale=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
            samples=torch.tensor(np.tile(points[1],(12,1)))
            targets=torch.tensor(np.tile(target[1],(12,1)))
            loss=unsigned_pad_loss(samples*torch.exp(logscale)-targets,float(weights[1]))
            loss.backward()
            self.assertGreater(logscale.grad.item(),0.)
            candidate_scale=1./(1.+gap)
            self.assertTrue(HAND_SCALE_BOUNDS[0] <= candidate_scale <= HAND_SCALE_BOUNDS[1])
            corrected=samples*candidate_scale
            self.assertLess(unsigned_pad_loss(corrected-targets,1.).item(),1e-10)
            np.testing.assert_allclose(project(points,np.array([600,600,300,300])),project(points*candidate_scale,np.array([600,600,300,300])))
        weights=visual_proximity_weights([True]*5,[0.]*5,[.05,np.nan,np.inf,-.1,.03],primary=[True]*5,reachable=[True]*5)
        self.assertEqual(np.flatnonzero(weights).tolist(),[4])

    def test_reachability_requires_one_shared_similarity_for_all_active_pads(self):
        points=np.array([[.5,.1,.5],[.5,-.1,.5],[.03,.04,.20],[.02,-.04,.20],[.01,.03,.20]])
        # Each point separately admits a scale/translation, but the two active
        # targets demand incompatible shared scales; no expanded gate is safe.
        target=points.copy(); target[0]*=1.18; target[1]*=.85
        independent=primary_similarity_reachable(points,target,active=[True,False,False,False,False])
        self.assertTrue(independent[0])
        independent=primary_similarity_reachable(points,target,active=[False,True,False,False,False])
        self.assertTrue(independent[1])
        self.assertFalse(primary_similarity_reachable(points,target,active=[True,True,False,False,False]).any())

    def test_reachability_is_directional_not_unlimited_norm_budget(self):
        points=np.tile([0.,0.,.1],(5,1))
        target=points+[.04,0.,0.]
        self.assertFalse(primary_similarity_reachable(points,target).any())
        target=points+[0.,0.,.04]
        self.assertTrue(primary_similarity_reachable(points,target).all())
        target=points+[0.,0.,.06]
        self.assertFalse(primary_similarity_reachable(points,target).any())
        target=points+[HAND_TRANSLATION_BOUND_M,0.,0.]
        self.assertTrue(primary_similarity_reachable(points,target).all())
        points[0]=np.nan;points[1,2]=-.1
        self.assertFalse(primary_similarity_reachable(points,target)[:2].any())


if __name__=='__main__':unittest.main()
