"""Disjoint image-group protocols. Unknown near-duplicates still require auditing."""
import hashlib

import numpy as np


def group_ids(paths, hashes=None, related=None):
    """Union equal paths, exact content hashes and curator-supplied related IDs."""
    paths = np.asarray(paths).astype(str)
    parent = list(range(len(paths)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values in (paths, hashes, related):
        if values is None:
            continue
        if len(values) != len(paths):
            raise ValueError("Group metadata length mismatch")
        seen = {}
        for i, value in enumerate(values):
            value = str(value)
            if not value:
                continue
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i
    members = {}
    for i, path in enumerate(paths):
        identity = str(hashes[i]) if hashes is not None and hashes[i] else path
        members.setdefault(root(i), set()).add(identity)
    names = {r: hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()
             for r, v in members.items()}
    return np.array([names[root(i)] for i in range(len(paths))])


def grouped_split(groups, fractions=(.6, .15, .125, .125), seed=0):
    """Group-random split, not domain-stratified. Fractions refer to groups, not rows."""
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.ndim != 1 or not np.isfinite(fractions).all()
            or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1)):
        raise ValueError("Positive split fractions summing to one required")
    groups = np.asarray(groups)
    unique = np.random.default_rng(seed).permutation(np.unique(groups))
    cuts = np.cumsum(np.floor(fractions * len(unique)).astype(int))[:-1]
    chunks = np.split(unique, cuts)
    if any(not len(chunk) for chunk in chunks):
        raise ValueError("Too few groups for requested splits")
    return tuple(np.flatnonzero(np.isin(groups, chunk)) for chunk in chunks)


def independent_indices(labels, groups):
    """One representative per annotated component for IID metrics/calibration.

    Mixed-label components (e.g. real originals plus AI edits) require a separate
    grouped-evaluation protocol; fail instead of selecting a misleading label.
    """
    first = {}
    for i, group in enumerate(groups):
        if group in first and labels[first[group]] != labels[i]:
            raise ValueError("Mixed labels within an image group; curate edits separately")
        first.setdefault(group, i)
    return np.array(sorted(first.values()), dtype=int)


def logo_split(labels, sources, groups, held_out, seed=0):
    labels, sources, groups = map(np.asarray, (labels, sources, groups))
    target = (labels == 1) & (sources == held_out)
    if not target.any():
        raise ValueError("Held-out generator not present")
    target_groups = set(groups[target])
    real_groups = sorted(set(groups[labels == 0]) - target_groups)
    if len(real_groups) < 2:
        raise ValueError("LOGO needs at least two independent real groups")
    real_groups = np.random.default_rng(seed).permutation(real_groups)
    test_groups = target_groups | set(real_groups[:max(1, int(np.ceil(.2 * len(real_groups))))])
    test_member = np.isin(groups, list(test_groups))
    return (np.flatnonzero(~test_member & ~target),
            np.flatnonzero(target | ((labels == 0) & test_member)))
