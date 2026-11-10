from collections import namedtuple
from fractions import Fraction

import numpy as np
from scipy.special import logsumexp

from oxyformer.provenance import require
from oxyformer.validation.generators import _groups
from oxyformer.validation.scm import (
    SCMConfig, LatentState, adjustment_key, exact, wide, structural_mean,
    observation_log_probability,
)

OracleNuisances = namedtuple('OracleNuisances', 'original_ids mu_a mu_d r_a r_d')

class ObservedLawOracle:
    def __init__(self, observed, truth, policy):
        self.observed, self.policy = observed, policy
        self.frame = observed.frame
        self.config = SCMConfig.from_json(truth.generator_configuration)
        require(truth.observation_hash == observed.content_hash, "wrong observation hash")
        require(truth.frame_hash == self.frame.content_hash, "wrong frame hash")
        require(truth.config_hash == self.config.content_hash, "wrong config hash")
        require(truth.policy_id == policy.policy_id, "wrong policy")
        require(truth.status == 'integrated', "truth not integrated")
        fixed = {oid: a for oid, a, key in zip(self.frame.original_ids, observed.a, self.frame.support_keys)
                 if not dict(policy.components_by_key)[key]}
        self.config.validate_policy(policy, self.frame, fixed_exposures=fixed)
        self.excluded_ids = frozenset(fixed)
        self.excluded_fixed_exposure_rows = len(fixed)
        self.groups = _groups(self.frame, self.config, policy)
        self.rows = dict(zip(self.frame.original_ids, range(len(self.frame.original_ids))))

    def _terms(self, row):
        measured = dict(zip(self.observed.measured_columns, self.observed.measured_x[row]))
        state = LatentState(local=measured.get('local_confounder', 0.),
                            regional=measured.get('regional_confounder', 0.))
        key = adjustment_key(self.frame, row, state, self.config)
        require(key in self.groups, 'empty adjustment stratum')
        return self.groups[key]

    def _kernels(self, point, terms):
        entries = []
        for term in terms:
            for piece in term.law.pieces:
                if piece.lower+term.law.error <= point <= piece.upper+term.law.error:
                    entries.append((term, term.law.kernel_at(piece, point)))
                    break
        return entries

    def evaluate(self, oid, a):
        require(oid not in self.excluded_ids, 'excluded fixed-exposure training row')
        row = self.rows[oid]
        terms = self._terms(row)
        point = exact(a)
        key = self.frame.support_keys[row]
        components = dict(self.policy.components_by_key)[key]
        delta = exact(self.policy.delta_mmhg)
        def moves(a):
            return delta != 0 and any(exact(lo) <= a <= exact(hi)-delta for lo, hi in components)
        factual = self._kernels(point, terms)
        require(bool(factual), 'outside support')
        incoming = self._kernels(point-delta, terms) if moves(point-delta) else []
        ref = max(k for _, k in factual+incoming)
        def logs(entries):
            return np.array([wide(k-ref)+t.log_weight-t.law.log_normalizer
                             +observation_log_probability(float(point), t.state, self.config)
                             for t, k in entries], dtype=np.longdouble)
        denom = logs(factual)
        log_denom = logsumexp(denom)
        weights = np.exp(denom-log_denom)
        means = [structural_mean([wide(point-t.law.error)], self.frame, t.row, t.state, self.config)[0]
                 for t, _ in factual]
        mean = np.sum(weights*np.asarray(means, dtype=np.longdouble))
        numerator = list(logs(incoming))
        if not moves(point):
            numerator.extend(denom)
        ratio = np.exp(logsumexp(numerator)-log_denom) if numerator else np.longdouble(0)
        require(bool(np.isfinite(mean) and np.isfinite(ratio)), 'nonfinite oracle')
        return mean, ratio

    def nuisances(self, original_ids):
        original_ids = tuple(i for i in original_ids if i not in self.excluded_ids)
        values = []
        for oid in original_ids:
            row = self.rows[oid]
            a = self.observed.a[row]
            moved = self.policy.shift_mask([a], [self.frame.support_keys[row]])[0]
            d = exact(a)+(exact(self.policy.delta_mmhg) if moved else Fraction(0))
            qa, ra = self.evaluate(oid, a)
            qd, rd = self.evaluate(oid, d)
            values.append((float(qa), float(qd), float(ra), float(rd)))
        return OracleNuisances(tuple(original_ids), *(tuple(v[i] for v in values) for i in range(4)))
