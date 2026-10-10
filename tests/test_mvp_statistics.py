import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('validate_local_mvp',Path(__file__).parents[1]/'scripts/validate_local_mvp.py')
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def test_block_ci_constant_effect():
    d=np.full((12,3),-.02); mask=np.ones((12,3),dtype=bool)
    result=v.blocked_ci(d,mask,np.repeat(['a','b','c'],4),draws=100)
    assert result['dates']==3
    np.testing.assert_allclose(result['ci95'],[-.02,-.02])
    assert abs(result['delta_mae']+.02)<1e-12

def test_block_ci_weights_targets_not_date_means():
    d=np.array([[1.,1.,1.],[3.,3.,3.],[3.,3.,3.]])
    r=v.blocked_ci(d,np.ones_like(d,dtype=bool),np.array(['a','b','b']),draws=100)
    assert abs(r['delta_mae']-7/3)<1e-12
    assert r['ci95'][0]<=r['delta_mae']<=r['ci95'][1]
