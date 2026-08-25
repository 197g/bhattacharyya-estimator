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

// I disagree with clippy on enough of these to disable this for consistency.
#![allow(clippy::needless_range_loop)]
// Similarly here, at least right now I have readability concerns:
//
//     !(val >= 1.0)
//     val.partial_cmp(&1.0).is_none_or(Ordering::is_le)
//
// That seems rather awfully verbose. Open to suggestions here.
#![expect(clippy::neg_cmp_op_on_partial_ord)]

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
/// ```text
/// sum(l² · p_i + l · 2sqrt(p_i · minrange_i), i=j..=k)
///     <= maxrange{j,k} - sum(minrange_i, i=j..=k)
/// ```
///
/// Which is a quadratic equation in `l` with coefficients:
///
/// ```text
/// a = sum(p_i, i=j..=k)
/// b = sum(2sqrt(p_i · minrange_i), i=j..=k)
/// c = maxrange{j,k} - sum(minrange_i, i=j..=k)
/// ```
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
    if sorted.is_empty() {
        return ConstraintEstimator {
            estimate: super::Estimate::unknown(),
            distributed: 1.0,
        };
    }

    let count = sorted.len() as f64;
    let sqrt_n = count.sqrt();
    let expand = FloatVal::from_div(level.dkw_constant, sqrt_n).above;
    apply_for_expanded(expand, sorted, cdf)
}

pub(crate) fn apply_for_expanded(
    expand: f64,
    sorted: &[f64],
    cdf: &dyn super::ContinuousCDF<f64, f64>,
) -> ConstraintEstimator {
    fn upper_at(qs: &[f64], i: usize, expand: f64) -> f64 {
        FloatVal::sum_above([qs[i], expand]).min(1.0)
    }

    fn lower_at(qs: &[f64], i: usize, expand: f64) -> f64 {
        FloatVal::from_add_with_magnitude(qs[i], -expand, qs[i] > expand)
            .below
            .max(0.0f64)
    }

    let count = sorted.len() as f64;
    // Note: here we use a much smaller step than other estimators. The combined constraints take
    // care of ensuring that the error does not *add* up from this but rather the extra intervals
    // can be utilized. The only reason to reduce the data here is that the loop below is otherwise
    // cubic in runtime so we make it quadratic...
    let vstep = (count.powf(2.0 / 3.0) / count.ln().powf(1.0 / 3.0)).ceil() as usize;

    let mut qs: Vec<_> = (0..sorted.len())
        .step_by(vstep)
        .map(|n| (1. + n as f64) / count)
        .collect();

    // Square roots of `p_i` but convenient to initialize it with `p_i` itself.
    let mut ps: Vec<_> = sorted
        .chunks(vstep)
        .map(|arr| FloatVal::from_exact(cdf.cdf(arr[0])))
        .collect();
    ps.push(FloatVal::from_exact(1.0));

    let mut lowers = (0..qs.len())
        .map(|n| lower_at(&qs, n, expand))
        .collect::<Vec<_>>();
    lowers.push(1.0);

    let mut uppers = (0..qs.len())
        .map(|n| upper_at(&qs, n, expand))
        .collect::<Vec<_>>();
    uppers.push(1.0);

    let mut cdf_in_edf = true;
    for idx in 0..ps.len() {
        cdf_in_edf &= lowers[idx] <= ps[idx].above;
        cdf_in_edf &= ps[idx].below <= uppers[idx];
    }

    FloatVal::diff_cdf_in_place(&mut ps);

    ps.iter_mut().for_each(|p_i| {
        assert!(
            p_i.below >= 0.0 && p_i.above >= 0.0,
            "Quantiles must be non-decreasing, something is wrong"
        );
    });

    // After expansion, fix this up. I.e. we have a slack variable at the end for anything after the
    // last sample and this must capture all remaining distribution.
    qs.push(1.0);

    let minrange = (0..qs.len())
        .map(|n| {
            if n == 0 {
                FloatVal::from_exact(lowers[0])
            } else {
                // Make sure we only relax this constraint `p_i >= lower[i] - upper[i - 1]` for all
                // intervals and their associated variable..
                //
                // FIXME: If you have dense enough sampling this inversion is not that unlikely
                // to occur by overlapping intervals around consecutive CDF samples. See this
                // vertical diagram of a CDF and confidence correction particularly at the end or
                // start of a short-tail distribution:
                //
                // ```
                // n-1: |-x-|
                // n  :   |-x-|
                // ```
                //
                // Now, we only relax the constraint but we do that permanently. Instead, we could
                // still track the current bound like normal, just relaxing it in each individual
                // evaluation instead (e.g. `c.max(0.0)` in prefix_sum_iterator) and other places.
                // Note that we already do `FloatVal::sum_above` to setup the solver itself that
                // deals with the same basic issue over inverted interval bounds.
                let v = FloatVal::from_add_with_magnitude(
                    lowers[n].max(uppers[n - 1]),
                    -uppers[n - 1],
                    false,
                );

                FloatVal {
                    above: v.above.max(0.0),
                    below: v.below.max(0.0),
                }
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
    // Here we need an underestimation to initialize the constraint system.
    let raw_a: Vec<_> = ps.iter().map(|ival| ival.below).collect();

    // this will become at least: 2.0 * sqrt(minrange * p_i).
    let raw_b: Vec<_> = (0..raw_a.len()).map(|_| 0.0).collect();

    // `raw_c` is the compensation for `offset`. From the constraints `s_i >= ival` we subsequently
    // update these whenever we update any of the variables in that sum. In the general case:
    //
    // `a·s_i² + b·s_i >= ival - c`
    //
    // where `c` captures the constants moved to the right side. Initially, we have the offset
    // `-sqrt(ival)²` from the substitution initializing `offset` below.
    let raw_c: Vec<_> = minrange.iter().map(|_| 0.0).collect();

    for c in &raw_c {
        assert!(
            *c >= 0.0,
            "Invalid initial constraint bound, should be dropped?"
        );
    }

    // From now on we need sqrt(p_i) for the gradient direction and other coefficients. We no
    // longer need the original `p_i` for much so let's reuse that allocation.
    let sqrtp: Vec<_> = ps.into_iter().map(Interval::new).collect();

    // Track each variable's offset, i.e. sum up the steps we take while it is active.
    let mut offset: Vec<_> = minrange.iter().map(|_| 0.0).collect();

    assert!(
        {
            let mut sum = FloatVal::from_exact(0.0);

            for i in 0..sqrtp.len() {
                sum = sum + FloatVal::sum_above([uppers[i], -lowers_pre_j[i]]);
            }

            sum.above >= 1.0
        },
        "Single constraint-solution would not add to > 1.0"
    );

    assert!(
        {
            let mut sum = FloatVal::from_exact(0.0);

            for ival in &sqrtp {
                sum = sum + FloatVal::from_mul(ival.sqrt.above, ival.sqrt.above).above;
            }

            sum.above >= 1.0
        },
        "Self-solution would not add to > 1.0"
    );

    let mut pre = PrefixLookup {
        active: core::iter::repeat_n(true, qs.len()).collect(),
        a: raw_a,
        b: raw_b,
        c: raw_c,
    };

    let mut total_interval = 0.0;
    let mut total_p = 0.0;

    let mut value = 0.0;

    while !pre.is_empty() {
        let mut lambda = f64::INFINITY;
        let mut best = (0, 0, [0.0, 0.0, 0.0]);

        assert_eq!(pre.active.len(), sqrtp.len());
        assert_eq!(pre.a.len(), sqrtp.len());
        assert_eq!(pre.b.len(), sqrtp.len());
        assert_eq!(pre.c.len(), sqrtp.len());
        assert_eq!(offset.len(), sqrtp.len());

        for (j, k, [a, b, prec]) in pre.prefix_sum_iterator() {
            let ival = FloatVal::sum_above([uppers[k], -lowers_pre_j[j]]);
            let c = FloatVal::sum_above([ival, -prec]);

            assert!(
                c >= 0.0,
                "c must be an overestimation of a non-negative number"
            );

            let a_max = solve(a, b, c);
            assert!(a_max >= 0.0);

            if a_max < lambda {
                lambda = a_max;
                best = (j, k, [a, b, prec]);
            }
        }

        if !lambda.is_finite() {
            assert!(lambda > 0.0);
            break;
        }

        assert!(lambda >= 0.0);
        // Remove (j..k) from the problem and update the prefix sums.
        let (j, k, _debug_setup) = best;

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

        // Then update variable offsets themselves. Note that we need to overestimate all variables
        // in the end, consequently this must error on the side of stepping further than `adjust`.
        for idx in 0..pre.a.len() {
            let p_i = &sqrtp[idx];
            let o = &mut offset[idx];

            // Only open variables.
            if pre.active[idx] {
                let step = FloatVal::from_mul(lambda, p_i.sqrt.above);
                *o = FloatVal::sum_above([*o, step.above]);
            }
        }

        // Count contribution from open variables.
        for idx in j..=k {
            let p_i = &sqrtp[idx];
            let o = &mut offset[idx];

            if pre.active[idx] {
                let coeff_i = p_i;
                let r_i = *o;

                // These are just debugging..
                total_interval = r_i.mul_add(r_i, total_interval);
                total_p = FloatVal::from_add(total_p, coeff_i.len.above).above;

                // Value gives us the final result, we also must overestimate it.
                // FIXME: evaluate if recording all contributions separately with only a final
                // exact-sum step is actually more beneficial.
                value = {
                    let contribution = FloatVal::from_mul(coeff_i.sqrt.above, r_i).above;
                    FloatVal::from_add(contribution, value).above
                };
            }
        }

        /*
        eprintln!(
            "Step length: {lambda}/{:?}, value: {value} / {total_p}×{total_interval}",
            j..=k
        );
        */

        pre.remove(j, k, &offset);

        if cfg!(debug_assertions) {
            let mut ival = FloatVal::from_exact(0.0);
            let mut cval = 0.0;

            for idx in j..=k {
                let c = pre.c[idx];
                let o = offset[idx];

                cval = FloatVal::from_add(cval, c).below;
                ival = ival + FloatVal::from_mul(o, o).above;
            }

            assert!(
                ival.above >= cval,
                "Constraints do not underestimate variables {}/{:?}\n{_debug_setup:?}",
                cval,
                ival,
            );
        }

        if cfg!(debug_assertions) {
            let mut ival = FloatVal::from_exact(0.0);
            let mut cval = 0.0;

            for idx in j..=k {
                let c = pre.c[idx];
                let o = offset[idx];

                cval = FloatVal::from_add(cval, c).below;
                ival = ival + FloatVal::from_mul(o, o).above;
            }

            assert!(
                FloatVal::sum_above([uppers[k], -lowers_pre_j[j]]) <= ival.above,
                "Variables do not overfill whole constraint: !({} <= {}) (constraint: {})\n{_debug_setup:?}\nvar: {:?}\nc: {:?}\nps{:?}",
                FloatVal::sum_above([uppers[k], -lowers_pre_j[j]]),
                ival.above,
                cval,
                &offset[j..=k],
                &pre.c[j..=k],
                &sqrtp[j..=k],
            );
        }
    }

    // We may have intervals so small that the simultaneous step length by solving is infinite.
    // This, problematically, would leave some data unaccounted for. We must instead overestimate
    // the contribution from those variables individually.
    for idx in 0..pre.a.len() {
        let p_i = &sqrtp[idx];
        let o = &mut offset[idx];

        if pre.active[idx] {
            let coeff_i = p_i;

            // Fallback: make these variables take up their whole range. In this manner they might
            // violate some combined constraint but that's the error side we allow.
            let ival = FloatVal::sum_above([uppers[idx], -lowers_pre_j[idx]]);
            let r_i = (*o).max(FloatVal::from_sqrt(ival).above);
            *o = r_i;

            // These are just debugging..
            total_interval = r_i.mul_add(r_i, total_interval);
            total_p = FloatVal::from_add(total_p, coeff_i.len.above).above;

            // Value gives us the final result, we also must overestimate it.
            value = {
                let contribution = FloatVal::from_mul(coeff_i.sqrt.above, r_i).above;
                FloatVal::from_add(contribution, value).above
            };
        }
    }

    assert!(total_p >= 1.0, "All data accounted for");

    {
        let mut sum = FloatVal::from_exact(0.0);

        for &r_i in &offset {
            sum = sum + FloatVal::from_mul(r_i, r_i).above;
        }

        assert!(
            sum.above >= 1.0,
            "Self-solution of distribution would not add to >= 1.0, is {sum:?}"
        );
    }

    if cdf_in_edf && !(value >= 1.0) {
        // This branch will error. So give detailed info.
        for idx in 0..sqrtp.len() {
            eprintln!(
                "{}/{:?} -> {}",
                (sqrtp[idx].sqrt.above <= offset[idx]),
                sqrtp[idx],
                offset[idx]
            );
        }

        let mut sum = FloatVal::from_exact(0.0);
        for idx in 0..sqrtp.len() {
            sum = sum + FloatVal::from_mul(offset[idx], sqrtp[idx].sqrt.above).above;
        }

        panic!(
            "An empirical distribution covering the CDF must allow for a perfectly matching solution but: {}/{:?}",
            value, sum
        );
    }

    // Note: we have covered _at most_ the whole interval. There may be missing spots since we
    // never assign any value to intervals with `p_i = 0` (those do not contribute to the value but
    // make the solution ill-defined).
    // eprintln!("Value at optimum: {value} / {total_p}×{total_interval}");
    // eprintln!("CDF match: {cdf_in_edf}");

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
            debug_assert!(b >= 0.0);
            return f64::INFINITY;
        }

        assert!(b >= 0.0, "Negative b coefficient in linear equation {b}");
        assert!(c >= 0.0, "Negative c coefficient in linear equation {c}");
        return c / b;
    }

    assert!(a >= 0.0, "Negative a coefficient in quadratic equation {a}");
    assert!(c >= 0.0, "Negative c coefficient in quadratic equation {c}");

    // Avoid cancellation issues. We need to estimate `ba` in both directions, once for the square
    // root term (we only care about the positive solution) and once for its contribution in the
    // rest.
    let ba = FloatVal::from_div(b, a);
    let ca = FloatVal::from_div(c, a).above;

    if !ba.above.is_finite() || !ca.is_finite() {
        if b <= 0.0 {
            debug_assert!(b >= 0.0);
            return f64::INFINITY;
        } else {
            // Discard the quadratic term `a`, approximately.
            return FloatVal::from_div(c, b).above;
        }
    }
    // Note: c was the negative of the usual constant term
    let d = (FloatVal::from_mul(ba.above, ba.above) + 4.0 * ca).above;

    if d < 0.0 {
        debug_assert!(false, "Negative discriminant in quadratic equation");
        return 0.0;
    }

    assert!(
        d.sqrt() >= b,
        "Negative solution in quadratic equation {a}x²-{c}"
    );

    // `ba.above**2` might overflow floating point range.
    let dsq = if d.is_finite() {
        FloatVal::from_sqrt(d).above
    } else {
        // Approximate this square root by `ba.above + sqrt(4.0 * ca)` instead.
        FloatVal::sum_above([ba.above, 2.0 * FloatVal::from_sqrt(ca).above])
    };

    // FIXME: subnormals might actually round this wrong? Unsure.
    FloatVal::sum_above([dsq, -ba.below]) / 2.0
}

#[derive(Default, Debug)]
struct Interval {
    len: FloatVal,
    sqrt: FloatVal,
}

impl Interval {
    fn new(v: FloatVal) -> Self {
        Interval {
            len: v,
            sqrt: FloatVal {
                above: FloatVal::from_sqrt(v.above).above,
                below: FloatVal::from_sqrt(v.below).below,
            },
        }
    }
}

struct PrefixLookup {
    active: Vec<bool>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl PrefixLookup {
    fn is_empty(&self) -> bool {
        !self.active.iter().copied().any(|x| x)
    }

    /// Adjust the constraint system with a known step.
    ///
    /// Note we must only ever *relax* the constraints. That is, the right side is only allowed to
    /// get larger than its real value, and `b` smaller. `c` is the right side compensation
    /// subtracting the constant terms accumulated from substitutions and hence we also error on the
    /// lower side.
    fn adjust(&mut self, sq: &[Interval], lambda: f64) {
        // Redefine s'_i = s_i - lambda * sq_i. This is the substitution of variables after taking a
        // step in the direction of the gradient. (Recall: a_i² = sq_i.len)
        let c_step = FloatVal::from_mul(lambda, lambda);

        assert_eq!(self.a.len(), sq.len());
        assert_eq!(self.b.len(), sq.len());
        assert_eq!(self.c.len(), sq.len());

        for idx in 0..sq.len() {
            if !self.active[idx] {
                continue;
            }

            let c = &mut self.c[idx];
            let b = self.b[idx];
            let sq_i = &sq[idx];

            let lambda = FloatVal::from_exact(lambda);
            let sq_i_len = FloatVal::from_mul(sq_i.sqrt.below, sq_i.sqrt.below).below;

            // Note: b here is already biased with the direction of the step sqrt(p_i).
            let constants = FloatVal::from_add((lambda * b).below, (c_step * sq_i_len).below);
            *c = FloatVal::from_add(*c, constants.below).below;

            assert!(
                !(*c < 0.0),
                "Negative c coefficient after adjustment {lambda:?}·{sq_i:?}"
            );
        }

        for idx in 0..sq.len() {
            if !self.active[idx] {
                continue;
            }

            let b = &mut self.b[idx];
            let sq_i = &sq[idx];

            let lambda = FloatVal::from_exact(lambda);
            let sq_i_len = FloatVal::from_mul(sq_i.sqrt.below, sq_i.sqrt.below).below;

            *b = FloatVal::from_add(*b, 2.0 * (lambda * sq_i_len).below).below;

            assert!(
                *b >= 0.0,
                "Negative b coefficient after adjustment {lambda:?}·{sq_i:?}"
            );
        }
    }

    fn remove(&mut self, j: usize, k: usize, vars: &[f64]) {
        self.a[j..=k].fill(0.0);
        self.b[j..=k].fill(0.0);
        self.active[j..=k].fill(false);

        for (c, &var) in self.c[j..=k].iter_mut().zip(vars) {
            *c = (*c).min(FloatVal::from_mul(var, var).below);
        }
    }

    fn prefix_sum_iterator(&self) -> impl Iterator<Item = (usize, usize, [f64; 3])> + '_ {
        let n = self.a.len();

        // TODO: performance wise intervals must contain at least on active variable but intervals
        // are not identified by their active variables (e.g. even with only 1 active 1..=2 and
        // 1..=3 may have different constraint effects). Optimizing this means discarding
        // intervals more efficiently than a simple test.
        (0..n).flat_map(move |j| {
            (j..n)
                .scan(([0.0; 3], false), move |(acc, any_active), k| {
                    let is_active = self.active[k];

                    let a = if is_active { self.a[k] } else { 0.0 };
                    let b = if is_active { self.b[k] } else { 0.0 };
                    let c = self.c[k];

                    assert!(a >= 0.0);
                    assert!(acc[0] + a >= 0.0);
                    *any_active |= is_active;

                    // Accumulate all coefficients. (Fun fact: ML generated auto-complete had
                    // originally messed this up and just never stored the accumulator back).
                    *acc = [
                        FloatVal::from_add(acc[0], a).below,
                        FloatVal::from_add(acc[1], b).below,
                        FloatVal::from_add(acc[2], c).below,
                    ];

                    Some((j, k, (*acc, *any_active)))
                })
                .filter_map(
                    |(j, k, (acc, any_active))| if any_active { Some((j, k, acc)) } else { None },
                )
        })
    }
}

/// Represents an algebraic value bounded by two known floating point values.
///
/// Note: only represents non-negative values.
#[derive(Clone, Copy, Debug)]
struct FloatVal {
    below: f64,
    above: f64,
}

impl FloatVal {
    pub fn sum_above<const N: usize>(v: [f64; N]) -> f64 {
        v.iter()
            .fold(0.0, |c, &v| {
                FloatVal::from_add_with_magnitude(c, v, c.abs() > v.abs()).above
            })
            .max(0.0)
    }

    pub fn diff_cdf_in_place(slice: &mut [Self]) {
        let mut state = FloatVal::from_exact(0.0);
        slice.iter_mut().for_each(|x| {
            let pre = core::mem::replace(&mut state, *x);
            assert!(pre.above <= x.above, "Monotonicity of CDF");
            assert!(pre.below <= x.below, "Monotonicity of CDF");

            *x = FloatVal {
                above: FloatVal::from_add_with_magnitude(x.above, -pre.below, true).above,
                below: FloatVal::from_add_with_magnitude(x.below, -pre.above, x.below > pre.above)
                    .below,
            };
        })
    }

    pub fn from_exact(f: f64) -> Self {
        FloatVal { above: f, below: f }
    }

    pub fn from_sqrt(f: f64) -> Self {
        let ieee_rounded = f.sqrt();
        let bias = ieee_rounded.mul_add(ieee_rounded, -f);

        // sqrt*(x)**2 - x > 0
        if bias > 0.0 {
            FloatVal {
                above: ieee_rounded,
                below: ieee_rounded.next_down(),
            }
        } else {
            FloatVal {
                above: ieee_rounded.next_up(),
                below: ieee_rounded,
            }
        }
    }

    /// Get a Float representing an upper-bound on the product of these two numbers.
    ///
    /// Both must be non-negative.
    pub fn from_mul(lhs: f64, rhs: f64) -> Self {
        let naive = lhs * rhs;
        let err = lhs.mul_add(rhs, -naive);

        if err > 0.0 {
            FloatVal {
                above: naive,
                below: naive.next_down(),
            }
        } else {
            FloatVal {
                above: naive.next_up(),
                below: naive,
            }
        }
    }

    pub fn from_div(num: f64, denom: f64) -> Self {
        assert!(!(num < 0.0));
        assert!(denom > 0.0);

        let naive = num / denom;
        let err = naive.mul_add(denom, -num);

        if err >= 0.0 {
            FloatVal {
                above: naive,
                below: naive.next_down(),
            }
        } else {
            FloatVal {
                above: naive.next_up(),
                below: naive,
            }
        }
    }

    #[track_caller]
    pub fn from_add(lhs: f64, rhs: f64) -> Self {
        assert!(!(lhs < 0.0));
        assert!(!(rhs < 0.0));
        Self::from_add_with_magnitude(lhs, rhs, lhs > rhs)
    }

    fn from_add_with_magnitude(lhs: f64, rhs: f64, lhs_greater: bool) -> Self {
        let naive = lhs + rhs;
        // We can order our floats, we use this variant of 2Sum.
        let (max, min) = core::hint::select_unpredictable(lhs_greater, (lhs, rhs), (rhs, lhs));
        let residual = naive - max; // `min - rounding adjustment of naive`, exactly.
        let error = min - residual;

        if error > 0.0 {
            FloatVal {
                below: naive,
                above: naive.next_up(),
            }
        } else if (min - residual) < 0.0 {
            FloatVal {
                below: naive.next_down(),
                above: naive,
            }
        } else {
            FloatVal {
                below: naive,
                above: naive,
            }
        }
    }
}

impl Default for FloatVal {
    fn default() -> Self {
        FloatVal::from_exact(0.0)
    }
}

impl core::ops::Mul<f64> for FloatVal {
    type Output = FloatVal;
    #[track_caller]
    fn mul(self, v: f64) -> Self {
        assert!(!(v < 0.0), "Unimplemented non-positive range");
        FloatVal {
            above: FloatVal::from_mul(self.above, v).above,
            below: FloatVal::from_mul(self.below, v).below,
        }
    }
}

impl core::ops::Add<f64> for FloatVal {
    type Output = FloatVal;
    #[track_caller]
    fn add(self, v: f64) -> Self {
        assert!(!(self.below < 0.0), "Unimplemented non-positive range");
        FloatVal {
            above: FloatVal::from_add(self.above, v).above,
            below: FloatVal::from_add(self.below, v).below,
        }
    }
}

#[test]
fn verify_float_val() {
    let v = FloatVal::from_exact(0.5) + 0.5;
    assert!(v.above >= 1.0);
    assert!(v.below >= 1.0);
}
