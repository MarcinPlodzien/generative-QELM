"""Check output-label semantics at boundaries and over both full grids."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qgen.output_labels import label_outputs
for axes in [[np.linspace(0,12,4096)],[np.linspace(-3,3,64)]*2]:
 points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(4096,-1);r=label_outputs(points,axes)
 assert np.array_equal(r['indices'],np.arange(4096));assert np.array_equal(r['bits']@(2**np.arange(11,-1,-1)),r['indices'])
 assert ''.join(map(str,r['bits'][5]))=='000000000101'
r=label_outputs([[-1.],[.5],[1.5],[4.]], [np.arange(4.)]);assert r['indices'].tolist()==[0,1,2,3]
print('PASS:1D/2D full-grid labels, MSB order, midpoint ties and endpoint clipping')
