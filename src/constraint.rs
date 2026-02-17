//! We rewrite the problem:
//!
//! ```text
//! maximize f(x) = sqrt(p_i) * sqrt(q_i)
//! s.th.
//!     sum(q_i, i=0..=k) <= upper_k
//!     sum(q_i, i=0..=k) >= lower_k
//! ```
//!
//! Drop the minimum constraint except on each individual variable.
//!
//! ```text
//! s.th.
//!    sum(q_i, i=j..=k) <= upper_k - lower_j
//!    q_i >= lower_i - upper_{i-1}
//!    q_i >= 0
//! ```
//!
//! Re-identify the variables: r_i = sqrt(q_i)
//!
//! ```text
//! maximize f(x) = sqrt(p_i) * r_i
//! s.th.
//!     sum(r_i^2, i=j..=k) <= upper_k - lower_j
//!     r_i >= sqrt(max(lower_i - upper_{i-1}, 0))
//! ```
//!
//! And again: minrange_i = max(lower_i - upper_{i-1}, 0),
//!     maxrange{j,k} = upper_k - lower_{j-1}
//!     s_i = r_i - sqrt(minrange_i)
//!
//! ```text
//! maximize f(x) = sqrt(p_i) * (s_i + sqrt(minrange_i))
//! s.th.
//!    sum((s_i + sqrt(minrange_i))^2, i=j..=k) <= maxrange{j,k}
//!    s_i >= 0
//! ```
//!
//! Which we simplify by contracting the constant term:
//!
//! ```text
//! maximize f(x) = sqrt(p_i) * s_i
//! s.th.
//!     sum(s_i^2 + 2·s_i·sqrt(minrange_i), i=j..=k) <= maxrange{j,k} - sum(minrange_i, i=j..=k)
//!     s_i >= 0
//! ```
//!
//! Alright now we got ourselves a convex optimization problem. And we can very easily find an
//! local optimum point where the second order condition holds. Note all coefficients of the
//! Lagrangian second derivative are non-negative so second order qualification is easy.
//! Additionally note strong slater condition is also easy: for all `s_i` that are constrained to
//! zero by the bounds we can substitute an equivalent equality constraint and for all other we
//! have an easy feasible point with region. So then we just move in the direction of the gradient
//! of unfulfilled constraints until we hit the boundary of the feasible region.
//!
//! I don't know, maybe I've convinced myself of something completely untrue here.

#[non_exhaustive]
pub struct ConstraintEstimator {
    pub estimate: super::Estimate,
    /// The amount of the sample covered in the estimate. Intervals that have a zero-probability in
    /// the reference continuous CDF are never covered.
    pub distributed: f64,
}

/// See module documentation.
///
/// FIXME: this method is mathematically correct, I think, but numerically it isn't a perfect
/// upper-bound estimator. That is of course unfortunate. It is much better than the unreliable
/// estimator from the other module.
///
/// Here we apply the successive iteration steps. Note that in each we traverse in direction
/// `Sqrt(p_i)` for all in-active constraints. So we're searching for the length of the step until
/// hitting a constraint. Substituting the direction into the constraint gives us a quadratic
/// equation in the step length `l`. Let us assume `s_i = 0` for all `i` at the start of the iteration.
/// Then the constraint is:
///
///     sum(l² · p_i + l · 2sqrt(p_i · minrange_i), i=j..=k)
///         <= maxrange{j,k} - sum(minrange_i, i=j..=k)
///
/// Which is a quadratic equation in `l` with coefficients:
///     a = sum(p_i, i=j..=k)
///     b = sum(2sqrt(p_i · minrange_i), i=j..=k)
///     c = maxrange{j,k} - sum(minrange_i, i=j..=k)
///
/// Simple prefix sum problem if we want random access to the equations of each `j, k`. After
/// taking a step in the direction we can apply another substitution of variables (removing all
/// those covered by the constraint) and repeat until there are no more free variables (and no more
/// constraints). Note we're only interested in the value at the optimum so just track that.
pub fn apply(
    level: &super::ConfidenceLevel,
    sorted: &[f64],
    cdf: &dyn super::ContinuousCDF<f64, f64>,
) -> ConstraintEstimator {
    // FIXME: goes out of sync when we remove intervals of variables. We must make a copy.
    fn upper_at(qs: &[f64], i: usize, expand: f64) -> f64 {
        (qs[i] + expand).min(1.0)
    }

    fn lower_at(qs: &[f64], i: usize, expand: f64) -> f64 {
        (qs[i] - expand).max(0.0f64)
    }

    let count = sorted.len() as f64;
    let sqrt_n = count.sqrt();
    let expand = level.dkw_constant / sqrt_n;

    // Note: here we use a much smaller step than other estimators. The combined constraints take
    // care of ensuring that the error does not *add* up from this but rather the extra intervals
    // can be utilized. The only reason to reduce the data here is that the loop below is otherwise
    // cubic in runtime so we make it quadratic...
    let step = count.powf(2.0 / 3.0).ceil() as usize;

    let mut qs: Vec<_> = (0..sorted.len())
        .step_by(step)
        .map(|n| (1. + n as f64) / count)
        .collect();
    qs.push(1.0);

    // Square roots of `p_i` but convenient to initialize it with `p_i` itself.
    let mut ps: Vec<_> = sorted.chunks(step).map(|arr| cdf.cdf(arr[0])).collect();
    ps.push(1.0);
    super::Estimate::diff_in_place_with_added_bias(&mut ps, 0.0);
    ps.iter_mut().for_each(|p_i| {
        assert!(
            *p_i >= 0.0,
            "Quantiles must be non-decreasing, something is wrong"
        );
    });

    let mut lowers = (0..qs.len())
        .map(|n| lower_at(&qs, n, expand))
        .collect::<Vec<_>>();
    let uppers = (0..qs.len())
        .map(|n| upper_at(&qs, n, expand))
        .collect::<Vec<_>>();

    let minrange = (0..qs.len())
        .map(|n| {
            if n == 0 {
                0.0
            } else {
                (lowers[n] - uppers[n - 1]).max(0.0f64)
            }
        })
        .collect::<Vec<_>>();

    // No more need for the last lowers value.
    let lowers_pre_j = {
        lowers.insert(0, 0.0);
        lowers.pop();
        lowers
    };

    // First component of the constraint has `s_i²` which the step expands to `l²·sqrt(p_i)²`.
    let raw_a = ps.clone();
    let raw_b: Vec<_> = (0..raw_a.len())
        .map(|i| 2.0 * f64::sqrt(ps[i] * minrange[i]))
        .collect();
    let raw_c: Vec<_> = minrange;

    // From now on we need sqrt(p_i) for the gradient direction and other coefficients. We no
    // longer need the original `p_i` for much so let's reuse that allocation.
    ps.iter_mut().for_each(|p_i| *p_i = p_i.sqrt());
    let mut sqrtp = ps;

    // Recall we offset each `s_i` by its minimum value `sqrt(minrange_i)`. Use that to initialize
    // all the offsets which we later need to restore the actual value. Well; we don't *need* the
    // value itself if we keep track of the optimization value but this allows us to return the
    // argmax to the problem if asked (debugging) without large costs.
    let mut offset: Vec<_> = raw_c.clone();
    offset.iter_mut().for_each(|o| *o = o.sqrt());

    let mut pre = PrefixLookup {
        active: (0..qs.len()).collect(),
        a: raw_a,
        b: raw_b,
        c: raw_c,
    };

    let mut total_interval = 0.0;
    let mut total_p = 0.0;

    let mut value = 0.0;

    while !pre.is_empty() {
        let mut lambda = f64::INFINITY;
        let mut best = (0, 0);

        assert!(pre.active.len() <= sqrtp.len());
        assert_eq!(pre.a.len(), sqrtp.len());
        assert_eq!(pre.b.len(), sqrtp.len());
        assert_eq!(pre.c.len(), sqrtp.len());
        assert_eq!(offset.len(), sqrtp.len());

        for (j, k, [a, b, prec]) in pre.prefix_sum_iterator() {
            if a == 0.0 {
                continue;
            }

            let c = uppers[k] - lowers_pre_j[j] - prec;
            assert!(
                c >= 0.0,
                "Negative c coefficient in constraint {j}..{k}: {c}"
            );

            let a_max = solve(a, b, c);
            if a_max < lambda {
                lambda = a_max;
                best = (j, k);
            }
        }

        if !lambda.is_finite() {
            break;
        }

        assert!(lambda >= 0.0);
        // Remove (j..k) from the problem and update the prefix sums.
        let (j, k) = best;

        /*
        eprintln!(
            "Step length: {lambda}/{:?}, value: {value} / {total_p}×{total_interval}",
            j..=k
        );
        */

        // Note that we should be careful here, we want to remove variables from the problem but
        // the constraint system should stay defined as is. Rather, the constraint system is
        // collapsed with multiple sums now representing the same constraint. For instance, if we
        // look at constraint over 1..=2 and we remove variable 1, then we now only have a
        // constraint on 2..=2 with everything else being constants. The difference is crucial
        // because we want to use `lower_0`, not `lower_1` in the maxrange constraint.
        //
        // Update the prefix sums by removing the contribution of the removed variables. These are
        // unchanged from this point onwards.

        // First offset all the terms by the step (uses its current `b`).
        pre.adjust(&sqrtp, lambda);

        // Then update variable offsets themselves.
        for (p_i, o) in sqrtp.iter().zip(offset.iter_mut()) {
            // Only open variables.
            if *p_i > 0.0 {
                *o += lambda * p_i;
                /* Mathematically, yes. Numerically, no. And we don't care about exceeding the
                * maximum bound directly as that makes the function value appear larger as long as
                * it does not cause other steps to be smaller than possible.
                  assert!(
                *o * *o <= uppers[idx] - lowers_jplus_one[idx],
                "{o} >= {} at index {idx}, something is wrong",
                uppers[idx] - lowers_jplus_one[idx]
                ); */
            }
        }

        // Count contribution from open variables.
        for idx in j..=k {
            let p_i = &mut sqrtp[idx];
            let o = &mut offset[idx];

            if !(*p_i == 0.0) {
                let coeff_i = core::mem::replace(p_i, 0.0);

                let r_i = *o;
                let maximum = uppers[idx] - lowers_pre_j[idx];
                let r_i = f64::sqrt(maximum).min(r_i);
                *o = r_i;

                total_interval = r_i.mul_add(r_i, total_interval);
                total_p = coeff_i.mul_add(coeff_i, total_p);
                value = coeff_i.mul_add(r_i, value);
            }
        }

        pre.remove(j, k, &offset);
    }

    // Note: we have covered _at most_ the whole interval. There may be missing spots since we
    // never assign any value to intervals with `p_i = 0` (those do not contribute to the value but
    // make the solution ill-defined).
    // eprintln!("Value at optimum: {value} / {total_p}×{total_interval}");

    ConstraintEstimator {
        estimate: super::Estimate::from_bhattarachya_coefficient(value),
        distributed: total_interval,
    }
}

/// Solve al² + bl <= c for the maximum l >= 0. Note that a, b, c are / should be all non-negative.
fn solve(a: f64, b: f64, c: f64) -> f64 {
    assert!(a.is_finite());
    assert!(b.is_finite());
    assert!(c.is_finite());

    if a == 0.0 {
        if b <= 0.0 {
            return f64::INFINITY;
        }

        assert!(b >= 0.0, "Negative b coefficient in linear equation {b}");
        assert!(c >= 0.0, "Negative c coefficient in linear equation {c}");
        return c / b;
    }

    assert!(a >= 0.0, "Negative a coefficient in quadratic equation {a}");
    assert!(c >= 0.0, "Negative c coefficient in quadratic equation {c}");

    // Avoid cancellation issues.
    let ba = b / a;
    let ca = c / a;

    // Note: c was the negative of the usual constant term
    let d = ba.mul_add(ba, 4.0 * ca);

    if d < 0.0 {
        debug_assert!(false, "Negative discriminant in quadratic equation");
        return 0.0;
    }

    assert!(
        d.sqrt() >= b,
        "Negative solution in quadratic equation {a}x²-{c}"
    );

    (d.sqrt() - ba) / 2.0
}

struct PrefixLookup {
    /// Ordered list of active variable indices.
    active: Vec<usize>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl PrefixLookup {
    fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    fn adjust(&mut self, ps: &[f64], lambda: f64) {
        // Redefine s'_i = s_i - lambda * p_i. This is the substitution of variables after taking a
        // step in the direction of the gradient.
        assert_eq!(self.a.len(), ps.len());
        assert_eq!(self.b.len(), ps.len());
        assert_eq!(self.c.len(), ps.len());
        for ((c, sp_i), b) in self.c.iter_mut().zip(ps.iter()).zip(&self.b) {
            // Note: b here is already biased with the direction of the step sqrt(p_i).
            *c += b * lambda + sp_i * sp_i * lambda * lambda;
            assert!(
                *c >= 0.0,
                "Negative c coefficient after adjustment {lambda}·{sp_i}, something is wrong"
            );
        }

        for (b, sp_i) in self.b.iter_mut().zip(ps.iter()) {
            *b += 2.0 * sp_i * sp_i * lambda;
            assert!(
                *b >= 0.0,
                "Negative b coefficient after adjustment {lambda}·{sp_i}, something is wrong"
            );
        }
    }

    fn remove(&mut self, j: usize, k: usize, offset: &[f64]) {
        self.a[j..=k].fill(0.0);
        self.b[j..=k].fill(0.0);

        for (c, &o) in self.c[j..=k].iter_mut().zip(&offset[j..=k]) {
            *c = o * o;
        }
    }

    fn prefix_sum_iterator(&self) -> impl Iterator<Item = (usize, usize, [f64; 3])> + '_ {
        let n = self.a.len();

        // TODO: performance wise intervals must contain at least on active variable but intervals
        // are not identified by their active variables (e.g. even with only 1 active 1..=2 and
        // 1..=3 may have different constraint effects). Optimizing this means discarding
        // intervals more efficiently than a simple test.
        (0..n).flat_map(move |j| {
            (j..n).scan([0.0; 3], move |acc, k| {
                let a = self.a[k];
                let b = self.b[k];
                let c = self.c[k];
                assert!(a >= 0.0);
                assert!(acc[0] + a >= 0.0);

                // Accumulate all coefficients. (Fun fact: ML generated auto-complete had
                // originally messed this up and just never stored the accumulator back).
                *acc = [acc[0] + a, acc[1] + b, acc[2] + c];
                Some((j, k, *acc))
            })
        })
    }
}
