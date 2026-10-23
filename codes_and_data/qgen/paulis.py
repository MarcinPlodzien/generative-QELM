"""Pauli strings as integer letter arrays, and their conjugation tables.

CONVENTION
    A Pauli string on N qubits is an int8 array of length N with letters
    0 = I, 1 = X, 2 = Y, 3 = Z; entry q is the Pauli on qubit q.
    A k-qubit gate U acting on qubits (q_1, ..., q_k) uses MSB-first ordering: q_1 is the
    most significant factor, so the local Pauli with letters (l_1, ..., l_k) is
    sigma_{l_1} (x) ... (x) sigma_{l_k} and has local code a = sum_j l_j 4^(k-j).

HEISENBERG TABLE
    For a gate U we store C[a, b] with  U^dag sigma_a U = sum_b C[a, b] sigma_b.
    C is real because U^dag sigma_a U is Hermitian. For a Clifford gate every row of C has a
    single nonzero entry (+1 or -1); for a non-Clifford gate rows have several entries, which is
    the branching of Pauli paths.
"""
import itertools

import numpy as np

LETTERS = "IXYZ"
SIGMA = np.array([[[1, 0], [0, 1]],
                  [[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)


def local_pauli(code, k):
    """Matrix of the k-qubit local Pauli with code a (most significant letter first)."""
    letters = [(code // 4 ** (k - 1 - j)) % 4 for j in range(k)]
    M = np.array([[1.0 + 0j]])
    for l in letters:
        M = np.kron(M, SIGMA[l])
    return M


def heisenberg_table(U):
    """C[a, b] = Tr(sigma_b U^dag sigma_a U) / 2^k  for a k-qubit unitary U."""
    d = U.shape[0]
    k = int(round(np.log2(d)))
    P = np.stack([local_pauli(a, k) for a in range(4 ** k)])
    conj = np.einsum("ij,ajk,kl->ail", U.conj().T, P, U)          # U^dag sigma_a U
    C = np.einsum("bji,aij->ab", P, conj) / d                      # Tr(sigma_b^dag X), sigma_b Hermitian
    if np.max(np.abs(C.imag)) > 1e-10:
        raise ValueError("Heisenberg table is not real: U is not unitary?")
    C = C.real
    C[np.abs(C) < 1e-13] = 0.0
    return C


def to_label(letters):
    """int letters -> 'XIZ...' string in MSB-first qubit order."""
    return "".join(LETTERS[int(l)] for l in letters)


def from_label(label):
    return np.array([LETTERS.index(c) for c in label], dtype=np.int8)


def weight(letters):
    return int(np.count_nonzero(letters))


# ----------------------------------------------------------------------------
# Readout sets (the measured observables)
# ----------------------------------------------------------------------------
def readout_strings(N, qubits=None, kind="z", max_k=1, homogeneous=False, min_k=1, pattern=None,
                    support="contiguous"):
    """List of readout Pauli strings (int8 arrays of length N) supported on `qubits`.

    kind
      'z'            Z-basis readout only (the default; one computational-basis measurement):
                     all Z-strings Z_i, Z_iZ_j, ... of weight 1..max_k (max_k=1: just <Z_i>)
      'z_contiguous' Z-strings on windows of 1..max_k adjacent qubits
      'contiguous'   all windows of k = 1..max_k adjacent qubits (within `qubits`), all 3^k
                     letter choices (or only XX..X, YY..Y, ZZ..Z if homogeneous=True)
      'all_pairs'    single-qubit strings plus every pair (i, j) of `qubits`
      'all_k'        every subset of size <= max_k
    """
    qubits = list(range(N)) if qubits is None else list(qubits)
    if kind == "pattern":
        # one fixed basis per qubit (pattern = 'XZYY...', one measurement setting); strings are the
        # products over supports of weight min_k..max_k (contiguous blocks or all subsets)
        letters = np.array(["IXYZ".index(c) for c in pattern], np.int8)
        if support == "contiguous":
            sups = [tuple(qubits[s:s + k]) for k in range(min_k, max_k + 1) for s in range(len(qubits) - k + 1)]
        else:
            sups = [c for k in range(min_k, max_k + 1) for c in itertools.combinations(qubits, k)]
        out = []
        for sup in sups:
            st = np.zeros(N, dtype=np.int8)
            st[list(sup)] = letters[list(sup)]
            out.append(st)
        return out
    if kind in ("z", "z_contiguous"):
        if kind == "z":
            supports = [c for k in range(min_k, max_k + 1) for c in itertools.combinations(qubits, k)]
        else:
            supports = [tuple(qubits[s:s + k]) for k in range(min_k, max_k + 1)
                        for s in range(len(qubits) - k + 1)]
        out = []
        for sup in supports:
            s = np.zeros(N, dtype=np.int8)
            s[list(sup)] = 3
            out.append(s)
        return out
    if kind == "contiguous":            # neighbouring blocks of weight min_k..max_k
        supports = [tuple(qubits[s:s + k]) for k in range(min_k, max_k + 1)
                    for s in range(len(qubits) - k + 1)]
    elif kind == "all_pairs":
        supports = [(q,) for q in qubits] + list(itertools.combinations(qubits, 2))
    elif kind == "all_k":
        supports = [c for k in range(min_k, max_k + 1) for c in itertools.combinations(qubits, k)]
    else:
        raise ValueError(f"unknown readout kind '{kind}'")
    out = []
    for sup in supports:
        choices = [(p,) * len(sup) for p in (1, 2, 3)] if homogeneous else \
            itertools.product((1, 2, 3), repeat=len(sup))
        for lets in choices:
            s = np.zeros(N, dtype=np.int8)
            s[list(sup)] = lets
            out.append(s)
    return out
