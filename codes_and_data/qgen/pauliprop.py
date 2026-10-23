"""Heisenberg-picture Pauli propagation of a set of observables through a gate list.

For each input string P_m (m = 0..M-1) we compute the exact Pauli expansion

    U^dag P_m U = sum_t c_t Q_t ,

by conjugating with the gates in REVERSE order (U = G_n ... G_1, so U^dag P U =
G_1^dag ... G_n^dag P G_n ... G_1). All M observables are propagated together; a term is a row
(letters of Q_t, owner m) with coefficient c_t, and equal rows are merged after each gate.

Clifford gates keep one term per observable; a non-Clifford gate splits a term into at most
4^k. Terms with |c| < cutoff are dropped and their squared weight is accumulated in
`discarded`, so a caller can check how much was thrown away (it is 0 with cutoff=0).
"""
from dataclasses import dataclass

import numpy as np

from .paulis import heisenberg_table


@dataclass
class PauliSum:
    letters: np.ndarray     # (T, N) int8
    owner: np.ndarray       # (T,)  int32, which input observable the term belongs to
    coeff: np.ndarray       # (T,)  float64
    n_obs: int
    discarded: np.ndarray   # (n_obs,) squared coefficient weight dropped by the cutoff

    def n_terms_per_obs(self):
        return np.bincount(self.owner, minlength=self.n_obs)


def _merge(rows, coeff):
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    c = np.zeros(len(uniq))
    np.add.at(c, inv.reshape(-1), coeff)
    return uniq, c


def propagate(strings, gates, cutoff=1e-12, max_terms=2_000_000):
    """Heisenberg images U^dag P U of every string in `strings` (list/array of int8 letters)."""
    strings = np.asarray(strings, dtype=np.int8)
    M, N = strings.shape
    letters = strings.copy()
    owner = np.arange(M, dtype=np.int64)
    coeff = np.ones(M)
    discarded = np.zeros(M)
    tables = {}
    for U, qubits in reversed(gates):
        key = id(U)
        if key not in tables:
            tables[key] = heisenberg_table(U)
        C = tables[key]
        k = len(qubits)
        loc = np.zeros(len(letters), dtype=np.int64)
        for q in qubits:
            loc = loc * 4 + letters[:, q]
        rows_C = C[loc]                                          # (T, 4^k)
        t_idx, b_idx = np.nonzero(rows_C)
        new_c = coeff[t_idx] * rows_C[t_idx, b_idx]
        new_letters = letters[t_idx].copy()
        for j, q in enumerate(qubits):
            new_letters[:, q] = (b_idx // 4 ** (k - 1 - j)) % 4
        new_owner = owner[t_idx]
        if not _is_one_to_one(rows_C):
            key_rows = np.concatenate([new_letters.astype(np.int64), new_owner[:, None]], axis=1)
            key_rows, new_c = _merge(key_rows, new_c)
            new_letters = key_rows[:, :N].astype(np.int8)
            new_owner = key_rows[:, N]
            drop = np.abs(new_c) < cutoff
            if drop.any():
                np.add.at(discarded, new_owner[drop], new_c[drop] ** 2)
                keep = ~drop
                new_letters, new_owner, new_c = new_letters[keep], new_owner[keep], new_c[keep]
        letters, owner, coeff = new_letters, new_owner, new_c
        if len(letters) > max_terms:
            raise RuntimeError(f"Pauli sum exceeded max_terms={max_terms}; raise cutoff or reduce depth")
    return PauliSum(letters, owner.astype(np.int32), coeff, M, discarded)


def _is_one_to_one(rows_C):
    """True if every term maps to exactly one term (Clifford gate on these rows): no merging needed."""
    return bool(np.all(np.count_nonzero(rows_C, axis=1) == 1))
