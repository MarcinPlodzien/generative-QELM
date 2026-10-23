"""Training loss: weighted W2 squared or its 32-direction sliced approximation."""
import numpy as np
from numba import njit
@njit(cache=True)
def transport_1d(generated_y,qs,target_y,ps):
 """Exact squared W2 and output-coordinate gradient between two sorted weighted atom laws."""
 grad=np.zeros(len(generated_y));loss=0.;i=0;j=0;a=qs[0];b=ps[0]
 while i<len(generated_y) and j<len(target_y):
  mass=min(a,b);delta=generated_y[i]-target_y[j];loss+=mass*delta*delta;grad[i]+=2*mass*delta;a-=mass;b-=mass
  if a<=1e-15:
   i+=1
   if i<len(generated_y):a=qs[i]
  if b<=1e-15:
   j+=1
   if j<len(target_y):b=ps[j]
 return loss,grad

class DistributionLoss:
 """W2 in 1D; deterministic angular quadrature of sliced W2 in 2D.

Couplings are recomputed from output distributions at every call. This is not
pointwise regression to a prescribed latent-to-target transport map.
"""
 def __init__(self,y,p,directions=32):
  d=y.shape[1]
  self.dirs=np.ones((1,1)) if d==1 else np.column_stack([np.cos((np.arange(directions)+.5)*np.pi/directions),np.sin((np.arange(directions)+.5)*np.pi/directions)])
  self.yp=[];self.pp=[]
  for v in self.dirs:
   yy=y@v;order=np.argsort(yy);self.yp.append(yy[order]);self.pp.append(p[order])
 def __call__(self,generated_y,q):
  g=np.zeros_like(generated_y);value=0.
  for v,y,p in zip(self.dirs,self.yp,self.pp):
   z=generated_y@v;order=np.argsort(z);cost,grad=transport_1d(z[order],q[order],y,p);value+=cost;g[order]+=grad[:,None]*v
  return value/len(self.dirs),g/len(self.dirs)
