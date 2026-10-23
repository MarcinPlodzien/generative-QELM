"""Node PMFs of the targets and the result selection of the paper."""
import json
import numpy as np
from .io import CODE,ROOT,sha
TARGETS_1D=['five_peak','gamma','narrow_broad','unequal_mixture']
TARGETS_2D=['two_rings','two_moons','diagonal_cross','three_stripes','four_lobes']
TARGETS=TARGETS_1D+TARGETS_2D
BASE_MODELS=['expectation_identity','expectation_cluster']+[f'expectation_random{s}' for s in range(3)]+['shot_identity']+[f'shot_random{s}' for s in range(3)]
MODELS=['expectation_identity','expectation_cluster']+[f'expectation_random{s}' for s in range(10)]+['shot_identity','shot_cluster']+[f'shot_random{s}' for s in range(10)]
PAPER_1D=['five_peak','gamma','unequal_mixture']
HAAR_MODELS=[f'{mode}_haar_K{K}_s{seed}' for mode in ('expectation','shot') for K in (1,2,4,8,16) for seed in (0,1,2)]
MODELS+=HAAR_MODELS

def models_for_target(name,stage='all'):
 """Settings of the paper; narrow_broad and 2D expectation values use seeds 0..2."""
 if name not in TARGETS or stage not in ('all','1d'):raise ValueError((name,stage))
 extra=[]
 if name in PAPER_1D:extra=[f'{mode}_random{s}' for mode in ('expectation','shot') for s in range(3,10)]
 elif name in TARGETS_2D and stage=='all':extra=[f'shot_random{s}' for s in range(3,10)]
 if stage=='all' and name in PAPER_1D+TARGETS_2D:extra+=['shot_cluster']
 if stage=='all' and name in ('five_peak','gamma'):extra+=HAAR_MODELS
 return BASE_MODELS+extra

def settings(stage='all'):
 return [(name,model) for name in TARGETS for model in models_for_target(name,stage)]

def result_folder(name,model):
 if model not in models_for_target(name):raise ValueError((name,model))
 if '_haar_' in model:return CODE/f'results/haar_depth/{name}/{model}'
 if model.startswith('expectation'):
  return CODE/f'results/refined_{1 if name in TARGETS_1D else 2}d/{name}/{model}'
 return CODE/f'results/{name}/{model}'
def target(name):
 if name not in TARGETS:raise ValueError(name)
 src=CODE/f'targets/{name}/target.npz'
 with np.load(src) as a:raw=a['coordinates'];p=a['probabilities']
 definition=json.loads(src.with_name('definition.json').read_text())
 offset=np.asarray(definition['coordinate_offset']);scale=np.asarray(definition['coordinate_scale'])
 y=(raw-offset)/scale
 return dict(name=name,y=y,p=p,raw=raw,offset=offset,scale=scale,d=y.shape[1],
  source=str(src.relative_to(CODE)),source_sha256=sha(src),
  training_source_sha256=definition['original_training_target_sha256'])
