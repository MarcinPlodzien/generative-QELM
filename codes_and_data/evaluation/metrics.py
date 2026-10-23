"""MMD squared and total variation: evaluation only, never training objectives."""
import numpy as np
from scipy.spatial.distance import cdist
SIGMAS=(.2,.5,1.,2.,4.)
def total_variation(p,q):return float(np.abs(np.asarray(p)-np.asarray(q)).sum()/2)

def kernel_self(y_generated,q):
 result=np.zeros(5)
 for i in range(0,len(y_generated),256):
  for j in range(0,i+1,256):
   d=cdist(y_generated[i:i+256],y_generated[j:j+256],'sqeuclidean');w=q[i:i+256,None]*q[None,j:j+256];mult=1 if i==j else 2
   for k,s in enumerate(SIGMAS):result[k]+=mult*np.sum(w*np.exp(-d/(2*s*s)))
 return result

class KernelTarget:
 def __init__(self,t):
  self.y=t['y'];self.p=t['p'];self.d=t['d'];self.yy=kernel_self(self.y,self.p)
  if self.d==2:self.a=np.unique(self.y[:,0]);self.b=np.unique(self.y[:,1]);self.P=self.p.reshape(64,64)
 def score(self,y_generated,q):
  xx=kernel_self(y_generated,q);xy=np.zeros(5)
  for i in range(0,len(y_generated),256):
   for k,s in enumerate(SIGMAS):
    if self.d==1:values=np.exp(-cdist(y_generated[i:i+256],self.y,'sqeuclidean')/(2*s*s))@self.p
    else:
     kx=np.exp(-.5*((y_generated[i:i+256,0,None]-self.a)/s)**2);ky=np.exp(-.5*((y_generated[i:i+256,1,None]-self.b)/s)**2);values=np.sum((kx@self.P)*ky,axis=1)
    xy[k]+=q[i:i+256]@values
  return float(sum(xx+self.yy-2*xy))
