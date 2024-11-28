use estimated_hellinger::{ConfidenceLevel, Estimate};

use rand::{distributions::OpenClosed01, Rng as _};
use statrs::distribution::{ContinuousCDF, Normal};

#[test]
fn generate_sample() {
    let n = Normal::new(0.0, 1.0).unwrap();
    const COUNT: usize = 1_000_000;

    let mut v: Vec<f64> = rand::thread_rng()
        .sample_iter(OpenClosed01)
        .take(COUNT)
        .map(|x| n.inverse_cdf(x))
        .collect();

    v.sort_by(|a, b| a.total_cmp(b));

    // Against itself, we have a very very high BC.
    check_ecdf(&v, &n, 0.99);

    let above = Normal::new(0.5, 1.0).unwrap();

    let bc = optimal_square_hellinger((0.0, 1.0), (0.5, 1.0));
    check_ecdf(&v, &above, bc);
}

fn check_ecdf(v: &[f64], n: &dyn ContinuousCDF<f64, f64>, min_bc: f64) {
    // The bad confidence levels, and in particular none-confidence level, will happen to
    // underestimate the true one by a tiny bit sometimes. That's expected?
    let min_bc = 0.98 * min_bc;

    let max_hc = 1.0 - min_bc;

    let estimate = Estimate::from_ecdf(v, n);
    assert!(estimate.bc_estimate >= min_bc, "{estimate:?}");
    assert!(estimate.hc_squared < max_hc, "{estimate:?}");

    let e95 = ConfidenceLevel::P95.apply(v, n);
    assert!(e95.bc_estimate >= min_bc, "{estimate:?}");
    assert!(e95.hc_squared < max_hc, "{estimate:?}");

    let e99 = ConfidenceLevel::P99.apply(v, n);
    assert!(e99.bc_estimate >= min_bc, "{estimate:?}");
    assert!(e99.hc_squared < max_hc, "{estimate:?}");

    assert!(estimate.bc_estimate <= e95.bc_estimate);
    assert!(e95.bc_estimate <= e99.bc_estimate);
}

fn optimal_square_hellinger(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (m1, s1) = a;
    let (m2, s2) = b;

    let v1 = s1.sqrt();
    let v2 = s2.sqrt();

    ((2.0 * v1 * v2) / (s1 + s2)).sqrt() * (-0.25 * (m1 - m2).powi(2) / (s1 + s2)).exp()
}
