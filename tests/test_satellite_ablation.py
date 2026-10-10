import numpy as np
from cloud2watt.satellite_ablation import make_satellite,crop_stats,fit_transform

def test_crop_uses_center_only():
    a=np.full((3,128,128),1e6,dtype='float32'); a[:,48:80,48:80]=np.arange(3)[:,None,None]
    v=crop_stats(a)
    np.testing.assert_array_equal(v[:12],np.tile(np.arange(3),4));np.testing.assert_array_equal(v[12:],0)

def test_daylight_leaves_thermal_channel_unchanged():
    full=np.ones((2,105),dtype='float32'); elev=np.array([[-10,0,2.5,5],[-1,-1,-1,-1]],dtype='float32')
    result=make_satellite(full,None,elev,daylight=True)
    frame=result[:,:60].reshape(2,4,15)
    np.testing.assert_array_equal(frame[:,:,2::3],1)
    np.testing.assert_array_equal(frame[0,:,0],[0,0,.5,1]);np.testing.assert_array_equal(frame[1,:,0],0)
    np.testing.assert_array_equal(result[:,60:].reshape(2,3,15),np.diff(frame,axis=1))

def test_transform_fit_only_and_satellite_only_clip():
    rng=np.random.default_rng(4);base=rng.normal(size=(100,27)).astype('float32');sat=rng.normal(size=(100,105)).astype('float32');fit=np.arange(100)<70
    x,p=fit_transform(base,sat,fit,robust=True)
    base[~fit]=1e5;sat[~fit]=1e7
    y,q=fit_transform(base,sat,fit,robust=True)
    np.testing.assert_array_equal(p['center'],q['center']);np.testing.assert_array_equal(p['scale'],q['scale'])
    assert np.max(abs(y[:,27:]))<=5 and np.max(y[~fit,:27])>5

def test_constant_features_stay_finite():
    x,p=fit_transform(np.ones((5,27),dtype='float32'),np.ones((5,105),dtype='float32'),np.ones(5,bool),robust=True)
    assert np.isfinite(x).all();np.testing.assert_array_equal(x,0)
