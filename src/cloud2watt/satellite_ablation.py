"""Fixed satellite processing variants; fitted transforms never see evaluation rows."""
import numpy as np

VARIANTS={'daylight':(True,False,False),'robust':(False,True,False),'local32':(False,False,True),'combined':(True,True,True)}

def crop_stats(a):
    a=np.asarray(a,dtype=np.float32)
    if a.shape!=(3,128,128):raise ValueError('expected three 128x128 channels')
    a=a[:,48:80,48:80]
    v=np.concatenate([np.nanmean(a[:,:16,:16],axis=(1,2)),np.nanmean(a[:,:16,16:],axis=(1,2)),np.nanmean(a[:,16:,:16],axis=(1,2)),np.nanmean(a[:,16:,16:],axis=(1,2)),np.nanstd(a,axis=(1,2))])
    if not np.isfinite(v).all():raise ValueError('nonfinite local features; do not silently drop evaluation rows')
    return v.astype('float32')

def make_satellite(full,local,elevation,*,daylight=False,small=False):
    frame=(local if small else full[:,:60].reshape(-1,4,15)).copy()
    if daylight:
        gate=np.clip(elevation/5,0,1).astype('float32')
        # Each group of three holds VIS006, IR_016, IR_108.
        reflectance=np.flatnonzero(np.arange(15)%3!=2)
        frame[:,:,reflectance]*=gate[:,:,None]
    return np.concatenate([frame.reshape(len(frame),60),np.diff(frame,axis=1).reshape(len(frame),45)],axis=1).astype('float32')

def fit_transform(base,sat,fit,*,robust=False):
    raw=np.concatenate([base,sat],axis=1).astype('float32')
    center=raw[fit].mean(0); scale=raw[fit].std(0); scale[scale<1e-6]=1
    if robust:
        train=sat[fit]
        center[27:]=np.median(train,axis=0)
        iqr=(np.quantile(train,.75,axis=0)-np.quantile(train,.25,axis=0))/1.349
        fallback=train.std(0)
        iqr=np.where(iqr<1e-6,fallback,iqr); iqr[iqr<1e-6]=1
        scale[27:]=iqr
    params={'center':center,'scale':scale,'robust':bool(robust),'satellite_clip':5.0 if robust else None,'fit_only':True}
    return apply_transform(base,sat,params),params

def apply_transform(base,sat,params):
    raw=np.concatenate([base,sat],axis=1).astype('float32'); x=(raw-params['center'])/params['scale']
    if params['robust']:x[:,27:]=np.clip(x[:,27:],-params['satellite_clip'],params['satellite_clip'])
    if not np.isfinite(x).all():raise ValueError('nonfinite model input')
    return x.astype('float32')
