use estimated_hellinger::{ConfidenceLevel, Estimate};

use rand::{distributions::OpenClosed01, Rng as _};
use statrs::distribution::{ContinuousCDF, Normal};

#[test]
fn generate_sample() {
    let n = Normal::new(0.0, 1.0).unwrap();
    const COUNT: usize = 4_000_000;

    let mut v: Vec<f64> = rand::thread_rng()
        .sample_iter(OpenClosed01)
        .take(COUNT)
        .map(|x| n.inverse_cdf(x))
        .collect();

    v.sort_by(|a, b| a.total_cmp(b));

    // Against itself, we have a very very high BC. In particular the extremely high confidence
    // levels must /never/ (in a computation heat death sense) overestimate this.
    check_ecdf(&v, &n, 1.0);

    for (mean, stddev) in [
        (0.5, 1.0),
        (0.0, 2.0),
        (1.0, 1.0),
        (0.0, 8.0),
        (8.0, 1.0),
        (-8.0, 1.0),
    ] {
        let above = Normal::new(mean, stddev).unwrap();
        let affinity = normal_square_bc((0.0, 1.0), (mean, stddev));

        check_ecdf(&v, &above, affinity);
    }
}

fn check_ecdf(v: &[f64], n: &dyn ContinuousCDF<f64, f64>, actual_bc: f64) {
    // The bad confidence levels, and in particular none-confidence level, will happen to
    // underestimate the true one by a tiny bit sometimes. That's expected?
    //
    // We need a better way to *grind* these tests to verify if the confidence levels hold! That is
    // determine the variance of the test itself.

    let estimate = Estimate::from_ecdf(v, n);
    let e95 = ConfidenceLevel::P95.apply(v, n);
    let e95_constraint = ConfidenceLevel::P95
        .apply_constraint_maximizer(v, n)
        .estimate;
    let e99 = ConfidenceLevel::P99.apply(v, n);
    let e99_constraint = ConfidenceLevel::P99
        .apply_constraint_maximizer(v, n)
        .estimate;
    let e_always = ConfidenceLevel::from_magnitude(12.5).apply(v, n);
    let c_always = ConfidenceLevel::from_magnitude(12.5)
        .apply_constraint_maximizer(v, n)
        .estimate;

    eprintln!("Actual {}", actual_bc);
    // This is probably too low, at least quite likely.
    eprintln!("Unreliable {}", estimate.bc_estimate);
    // This **must** be reliable, 12.5 digits of confidence! The hard thing is that this will also
    // be quite a way above `1.0` (a non-sense BC) in quite a few cases until a lot of data was
    // gathered.
    eprintln!("e-12 Estimate {}", e_always.bc_estimate);
    eprintln!("Constraint {}", c_always.bc_estimate);
    // Eh. Will fail once in a while.
    eprintln!("P99 {}", e99.bc_estimate);
    // This is just numerically too low but mostly correct.
    eprintln!("Constraint {}", e99_constraint.bc_estimate);
    // Eh. Would fail every few runs of these tests (though DFW is not very tight in our use for
    // larger sample sizes).
    eprintln!("P95 {}", e95.bc_estimate);
    eprintln!("Constraint {}", e95_constraint.bc_estimate);
    eprintln!("");

    assert!(estimate.bc_estimate <= e95.bc_estimate);
    assert!(e95.bc_estimate <= e99.bc_estimate);
    assert!(e99.bc_estimate <= e_always.bc_estimate);
    assert!(e_always.bc_estimate >= actual_bc);
}

fn normal_square_bc(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (m1, s1) = a;
    let (m2, s2) = b;

    let v1 = s1 * s1;
    let v2 = s2 * s2;

    ((2.0 * s1 * s2) / (v1 + v2)).sqrt() * (-0.25 * (m1 - m2).powi(2) / (v1 + v2)).exp()
}
