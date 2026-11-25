"""Platform keys for tests that compare CPU numerics bit-exactly to recorded fixtures.

A CUDA build of torch always selects the 'cuda' record, whose values the
cluster verifier enforces exactly. A CPU build selects a key built from what
the runtime reports for the factors measured to change a fixture (ARC-5117):
the OpenBLAS kernel (core type) and version NumPy and SciPy load, the ATen
CPU capability, and the CPU model, which stands in for the MKL and oneDNN
kernels torch dispatches to (they choose by CPU and report no choice).
Environment variables never enter the key; a run that overrides MKL or oneDNN
dispatch keeps its platform's key and must match its values, and failure
messages list such overrides. On a recorded platform the comparison is exact. On
an unrecorded CPU platform a GitHub Actions run compares every value to each
recorded CPU platform within a declared tolerance and names the key in a
warning; anywhere else the test fails and names the missing key.
"""
import json
import math
import os
import platform
import re
import warnings

import pytest
import torch

# |observed - recorded| <= abs_tol + rel_tol * |recorded|
FLOAT64 = dict(rel_tol=1e-9, abs_tol=1e-12)
# float32 model outputs (one float32 ulp is 2**-23 ~ 1.2e-7 relative). Across five
# ci-cpu platform settings (ATen, OpenBLAS, MKL and oneDNN AVX2 emulations) and an
# SSE emulation the predictions differ by at most 4.3e-6 relative (about 36 ulp)
# and 1.6e-6 absolute. Hosted CPUs that cannot be reproduced on the cluster (for
# example MKL's paths on AMD) may spread further, so the bound is about 24 times
# the measured spread. A change in training
# moves them far more: a batch size of 8 instead of 256 moves them 0.63 relative.
FLOAT32 = dict(rel_tol=1e-4, abs_tol=1e-6)
DIGEST = re.compile(r'[0-9a-f]{64}')
# Prefixes of variables that can steer kernel selection; listed in failure messages only.
OVERRIDE_PREFIXES = ('ATEN_', 'DNNL_', 'MKL_', 'NPY_', 'OMP_', 'ONEDNN_', 'OPENBLAS_')


def blas_cores():
    """Sorted '<library>-<version>-<core type>' of every BLAS loaded by NumPy and SciPy."""
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401  (loads SciPy's own BLAS so the key does not depend on import order)
    from threadpoolctl import threadpool_info
    return sorted({f"{lib.get('internal_api')}-{lib.get('version')}-{lib.get('architecture')}"
                   for lib in threadpool_info() if lib.get('user_api') == 'blas'})


def cpu_model():
    try:
        with open('/proc/cpuinfo') as cpuinfo:
            for line in cpuinfo:
                if line.startswith('model name'):
                    return ' '.join(line.split(':', 1)[1].split())
    except OSError:
        pass
    return platform.processor() or 'unknown'


def kernel_overrides():
    return sorted(f'{name}={value}' for name, value in os.environ.items() if name.startswith(OVERRIDE_PREFIXES))


FACTORS = ('cpu_model', 'aten', 'blas')
# Each fixture keys on the factors measured to change it (ARC-5117). The float64
# estimators change with the OpenBLAS kernel and version, not with MKL or oneDNN
# dispatch; the model changes with ATen, MKL and oneDNN dispatch (the CPU model),
# not with the OpenBLAS kernel. The ATen capability and CPU model are kept in both.
ESTIMATES = ('cpu_model', 'aten', 'blas')
MODEL = ('cpu_model', 'aten')


def platform_key(factors=FACTORS):
    """'cuda' for a CUDA build, else 'cpu/<model>/aten=..[/blas=..]' over ``factors``."""
    if torch.version.cuda:
        return 'cuda'
    unknown = set(factors) - set(FACTORS)
    assert not unknown, f'unknown platform factors {sorted(unknown)}'
    parts = ['cpu']
    if 'cpu_model' in factors:
        parts.append(cpu_model())
    if 'aten' in factors:
        parts.append(f'aten={torch.backends.cpu.get_cpu_capability()}')
    if 'blas' in factors:
        parts.append(f"blas={','.join(blas_cores())}")
    return '/'.join(parts)


def assert_close(observed, reference, *, rel_tol, abs_tol, opaque=(), where='value'):
    """Compare recorded floats within abs_tol + rel_tol * |reference|; everything else exactly.

    Keys named in ``opaque`` hold digests of floating-point state, which no
    tolerance applies to: they are compared by count and digest form only.
    """
    if isinstance(reference, dict):
        assert isinstance(observed, dict) and sorted(observed) == sorted(reference), \
            f'{where}: keys {sorted(observed) if isinstance(observed, dict) else observed!r} != {sorted(reference)}'
        for key in reference:
            if key in opaque:
                assert len(observed[key]) == len(reference[key]) and all(
                    isinstance(x, str) and DIGEST.fullmatch(x) for x in observed[key]), \
                    f'{where}.{key}: expected {len(reference[key])} sha256 digests'
            else:
                assert_close(observed[key], reference[key], rel_tol=rel_tol, abs_tol=abs_tol, opaque=opaque, where=f'{where}.{key}')
    elif isinstance(reference, (list, tuple)):
        assert isinstance(observed, (list, tuple)) and len(observed) == len(reference), \
            f'{where}: length {len(observed) if isinstance(observed, (list, tuple)) else observed!r} != {len(reference)}'
        for i, (a, b) in enumerate(zip(observed, reference)):
            assert_close(a, b, rel_tol=rel_tol, abs_tol=abs_tol, opaque=opaque, where=f'{where}[{i}]')
    elif type(reference) is float:
        # Only recorded floats get a tolerance; identifiers and counts stay exact.
        assert type(observed) is float, f'{where}: {observed!r} is not a float like {reference!r}'
        assert math.isfinite(observed) and math.isfinite(reference), f'{where}: {observed!r} vs {reference!r}'
        assert math.fabs(observed - reference) <= abs_tol + rel_tol * math.fabs(reference), \
            f'{where}: {observed!r} differs from {reference!r} beyond rel_tol={rel_tol}, abs_tol={abs_tol}'
    else:
        assert type(observed) is type(reference) and observed == reference, f'{where}: {observed!r} != {reference!r}'


def check(label, *, factors, exact, records, values, references, tolerance, opaque=()):
    """Exact against records[platform_key(factors)], else the hosted tolerance path.

    ``records`` maps platform keys to the expected form of ``exact``;
    ``references`` maps recorded CPU keys to the expected form of ``values``.
    Returns the platform key and the path taken ('exact' or 'tolerance').
    """
    key = platform_key(factors)
    found = kernel_overrides()
    overrides = f'; kernel overrides set: {found}' if found else ''
    if key in records:
        # JSON text, so an int recorded as 1103 does not match 1103.0, nor 0.0 match -0.0.
        assert json.dumps(exact, sort_keys=True) == json.dumps(records[key], sort_keys=True), \
            f'{label}: differs from the recorded fixture for platform {key!r}{overrides}'
        return key, 'exact'
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.fail(f'{label}: no recorded fixture for platform {key!r}{overrides} '
                    f'(recorded: {sorted(records)}); record its values before testing on this platform')
    assert references, f'{label}: no recorded CPU reference for unrecorded platform {key!r}'
    for name, reference in sorted(references.items()):
        assert_close(values, reference, **tolerance, opaque=opaque, where=f'{label} on {key!r} vs {name!r}')
    warnings.warn(f'{label}: platform {key!r} is not recorded; compared within rel_tol={tolerance["rel_tol"]}, '
                  f'abs_tol={tolerance["abs_tol"]} to {sorted(references)}')
    return key, 'tolerance'
