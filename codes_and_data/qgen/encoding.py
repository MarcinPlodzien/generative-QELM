"""Product encoding of latent z; the head receives features x and emits y.

Bloch factors for R_z(phi) R_y(theta)|0>: (sin(theta)cos(phi),
sin(theta)sin(phi),cos(theta)). Paper runs use contiguous latent blocks.
"""
from dataclasses import dataclass
import numpy as np
@dataclass
class Encoding:
 N:int
 D:int
 freq_base:float=np.pi
 ratio:float=2.
 phi:float=np.pi/4
 layout:str='blocks'
 def _var_tier(self):
  i=np.arange(self.N)
  if self.layout=='blocks':
   size=int(np.ceil(self.N/self.D));return i//size,i%size
  if self.layout=='interleaved':return i%self.D,i//self.D
  raise ValueError(self.layout)
 def variable(self):return self._var_tier()[0]
 def frequencies(self):return self.freq_base*self.ratio**self._var_tier()[1]
