"""The identity, cluster layer and random Clifford reservoirs used in the paper."""
import pickle
import numpy as np
from .io import CACHE_DIR
I2=np.eye(2,dtype=complex)
H=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
S=np.diag([1,1j]).astype(complex)
CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
CZ=np.diag([1,1,1,-1]).astype(complex)
CIRCUITS={}



def _canonical(U):
    flat = U.reshape(-1)
    k = np.argmax(np.abs(flat) > 1e-6)
    V = U * (abs(flat[k]) / flat[k])
    V = np.round(V, 6) + (0.0 + 0.0j)        # +0 turns -0.0 into 0.0 so equal gates hash equal
    return V.tobytes()


def clifford_group_2q():
    path = CACHE_DIR / "clifford_group_2q.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    gens = [np.kron(H, I2), np.kron(I2, H), np.kron(S, I2), np.kron(I2, S), CNOT]
    group = {_canonical(np.eye(4)): np.eye(4, dtype=complex)}
    frontier = list(group.values())
    while frontier:
        new = []
        for U in frontier:
            for g in gens:
                V = g @ U
                key = _canonical(V)
                if key not in group:
                    group[key] = V
                    new.append(V)
        frontier = new
        if len(group) > 11520:
            raise RuntimeError('Clifford enumeration did not close (canonicalisation bug)')
    arr = np.stack(list(group.values()))
    assert len(arr) == 11520, len(arr)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(arr, f)
    return arr


def brickwork_bonds(N, depth):
    """Bonds of a depth-`depth` brickwork on an open chain: even bonds, odd bonds, even, ..."""
    return [[(i, i + 1) for i in range(k % 2, N - 1, 2)] for k in range(depth)]


def register_circuit(name):
    def deco(f):
        CIRCUITS[name] = f
        return f
    return deco


def build_circuit(name, N, seed=0, **params):
    if name not in CIRCUITS:
        raise KeyError(f"unknown circuit '{name}'; available: {sorted(CIRCUITS)}")
    return CIRCUITS[name](N, np.random.default_rng(seed), **params)

@register_circuit("identity")
def _identity(N, rng):
    """No entangler: features are the readout strings evaluated on the product state."""
    return []

@register_circuit("random_clifford")
def _random_clifford(N, rng, depth=None):
    """Brickwork of uniformly random two-qubit Cliffords, depth L (default 2N)."""
    group = clifford_group_2q()
    depth = 2 * N if depth is None else depth
    return [(group[rng.integers(len(group))], b) for layer in brickwork_bonds(N, depth) for b in layer]

@register_circuit("cz_h_layers")
def _cz_h_layers(N, rng, depth=1):
    """Structured Clifford: [CZ on all nearest-neighbour bonds, H on all qubits] x depth."""
    out = []
    for _ in range(depth):
        out += [(CZ, (i, i + 1)) for i in range(N - 1)]
        out += [(H, (q,)) for q in range(N)]
    return out
