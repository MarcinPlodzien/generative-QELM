"""Output contract: continuous head coordinate -> fixed bin index -> binary label.

These output labels are not the measured quantum bitstrings used as inputs.
The functions accept coordinates in the declared training/metric coordinates.
"""
import numpy as np

def label_outputs(coordinates,axes):
 """Quantize to nearest nodes (midpoint ties upward), C-order/MSB-first labels.

Returns continuous coordinates, quantized coordinates, per-axis indices,
integer flat indices and a matrix of bits. One axis gives the1D index directly;
two64-node axes give index=64*i+j and a12-bit output label.
"""
 y=np.asarray(coordinates,dtype=float);y=y.reshape(-1,len(axes))
 if not np.isfinite(y).all():raise ValueError('Head outputs must be finite')
 sizes=tuple(len(a) for a in axes);count=int(np.prod(sizes));N=count.bit_length()-1
 if count!=2**N:raise ValueError('The output grid must have a power-of-two number of bins')
 for a in axes:
  if len(a)<2 or not np.all(np.diff(a)>0):raise ValueError('Each grid axis must be strictly increasing')
 ij=np.column_stack([np.searchsorted((a[:-1]+a[1:])/2,y[:,i],side='right') for i,a in enumerate(axes)])
 indices=np.ravel_multi_index(ij.T,sizes);bits=((indices[:,None]>>np.arange(N-1,-1,-1))&1).astype(np.uint8)
 nodes=np.column_stack([np.asarray(a)[ij[:,i]] for i,a in enumerate(axes)])
 return {'continuous_coordinates':y,'quantized_coordinates':nodes,'axis_indices':ij,'indices':indices,'bits':bits}

def affine_head_outputs(features,W,b,axes):
 """Apply the saved raw-feature affine head, then produce both output labels."""
 return label_outputs(np.asarray(features)@np.asarray(W)+np.asarray(b),axes)


def quantized_law(y,q,t):
 axes=[np.unique(t['y'][:,j]) for j in range(t['d'])];labels=[np.searchsorted((a[:-1]+a[1:])/2,y[:,j],side='right') for j,a in enumerate(axes)]
 ids=labels[0] if t['d']==1 else labels[0]*len(axes[1])+labels[1]
 p=np.bincount(ids,weights=q,minlength=4096);return p
