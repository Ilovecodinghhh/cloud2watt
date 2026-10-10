import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from evaluate_spring_validation import ci,score

def test_constant_paired_difference_stays_constant():
 dates=np.repeat(np.array(['2021-04-%02d'%d for d in range(21,29)]),4)
 diff=np.full((len(dates),3),-.02);mask=np.ones_like(diff,dtype=bool);mask[0,0]=False
 for length in [1,3]:
  result=ci(diff,mask,dates,length)
  np.testing.assert_allclose(result['ci95'],[-.02,-.02]);assert result['dates']==8

def test_horizons_equal_weight_despite_missingness():
 e=np.array([[1.,3.],[1.,999.]])
 mask=np.array([[True,True],[True,False]])
 assert score(e,mask)==2.
