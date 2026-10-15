import unittest
import numpy as np
from preprocessing.relative_repair import fit_scale_curve

class RelativeRepairTests(unittest.TestCase):
 def test_rigid_chords_recover_relative_depth_trend(self):
  t=np.arange(150)/30;true=np.exp(-.09*t/t[-1]);edges=[]
  for a in range(0,140,5):
   for b in [a+5,min(a+15,149)]:edges.append(dict(source_position=a,target_position=b,log_scale_difference=float(np.log(true[b]/true[a])),weight=1.))
  s,r=fit_scale_curve(t,edges)
  self.assertEqual(s[0],1.);self.assertLess(abs(s[-1]-true[-1]),.015);self.assertFalse(r['additive_bias']);self.assertLessEqual(r['temporal_curve_parameters'],7)
 def test_absent_support_is_no_new_metric_observation(self):
  t=np.arange(10)/30;s,r=fit_scale_curve(t,[]);np.testing.assert_array_equal(s,np.ones(10))
 def test_relative_gauge_not_absolute_scale(self):
  t=np.arange(50)/30;edges=[dict(source_position=i,target_position=i+1,log_scale_difference=0.,weight=1.) for i in range(49)];s,r=fit_scale_curve(t,edges);np.testing.assert_allclose(s,1.);self.assertEqual(r['anchor_scale'],1.)
if __name__=='__main__':unittest.main()
