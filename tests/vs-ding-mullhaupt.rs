//! Compare with the estimator(s) from:
//!
//! Ding, R., & Mullhaupt, A. (2023). Empirical Squared Hellinger Distance Estimator and Generalizations to a Family of α-Divergence Estimators. Entropy, 25(4), 612. https://doi.org/10.3390/e25040612
use estimated_hellinger::ConfidenceLevel;

use rand::{distributions::OpenClosed01, Rng as _};
use statrs::distribution::{ContinuousCDF, Normal};

#[test]
fn compare_figure_3a() {
    eprintln!("Comparing figure 3a");
    compare_normal((0.0, 4.0), (1.0, 1.0));
    eprintln!("Comparing figure 3b");
    compare_normal((0.0, 1.0), (2.0, 1.0));
    eprintln!("Comparing figure 4a");
    compare_normal((0.0, 1.0), (0.01, 1.0));
    eprintln!("Comparing figure 4b");
    compare_exp(1.0, 2.0);

    eprintln!("Comparing figure 5a");
    compare_uniform((0.0, 1.0), (0.0, 2.0));
    eprintln!("Comparing figure 5b");
    compare_uniform((0.0, 1.0), (0.5, 1.5));

    eprintln!("Comparing figure 6a");
    compare_cauchy_normal((0.0, 1.0), (1.0, 1.0));
    eprintln!("Comparing figure 6b");
    compare_cauchy_normal((0.0, 1.0), (0.0, 1.0));
}

fn compare_normal(p: (f64, f64), q: (f64, f64)) {
    // N(0, 4) and N(1, 1) in the paper passes the square of the deviation. statrs expects the
    // deviation itself instead.
    let dist_p = Normal::new(p.0, p.1.sqrt()).unwrap();
    let dist_q = Normal::new(q.0, q.1.sqrt()).unwrap();
    let h2 = 1.0 - normal_square_bc((p.0, p.1.sqrt()), (q.0, q.1.sqrt()));

    eprintln!("True: {h2}");
    compare_samples(&dist_p, &dist_q);
}

fn compare_uniform(p: (f64, f64), q: (f64, f64)) {
    let slope_p = 1.0 / (p.1 - p.0);
    let slope_q = 1.0 / (q.1 - q.0);
    let overlap = (p.1.min(q.1) - p.0.max(q.0)).max(0.0);
    let h2 = 1.0 - overlap * (slope_p * slope_q).sqrt();

    let dist_p = statrs::distribution::Uniform::new(p.0, p.1).unwrap();
    let dist_q = statrs::distribution::Uniform::new(q.0, q.1).unwrap();

    eprintln!("True: {h2}");
    compare_samples(&dist_p, &dist_q);
}

fn compare_exp(p: f64, q: f64) {
    let dist_p = statrs::distribution::Exp::new(p).unwrap();
    let dist_q = statrs::distribution::Exp::new(q).unwrap();

    let h2 = 1.0 - (4.0 * p * q / (p + q).powi(2)).sqrt();
    eprintln!("True: {h2}");
    compare_samples(&dist_p, &dist_q);
}

fn compare_cauchy_normal(p: (f64, f64), q: (f64, f64)) {
    let dist_p = statrs::distribution::Cauchy::new(p.0, p.1).unwrap();
    let dist_q = statrs::distribution::Normal::new(q.0, q.1.sqrt()).unwrap();

    eprintln!("True: n/a (numerical)");
    compare_samples(&dist_p, &dist_q);
}

fn compare_samples(dist_p: &dyn ContinuousCDF<f64, f64>, dist_q: &dyn ContinuousCDF<f64, f64>) {
    for sample_size in [10, 32, 100, 316, 1_000, 3_162, 10_000, 31_622] {
        let mut v: Vec<f64> = rand::thread_rng()
            .sample_iter(OpenClosed01)
            .take(sample_size)
            .map(|x| dist_p.inverse_cdf(x))
            .collect();

        v.sort_by(|a, b| a.total_cmp(b));
        let (ours, surely) = check_ecdf(&v, dist_q);
        eprintln!("Sample size {sample_size}: {ours} {surely}");
    }
}

fn check_ecdf(v: &[f64], n: &dyn ContinuousCDF<f64, f64>) -> (f64, f64) {
    // The bad confidence levels, and in particular none-confidence level, will happen to
    // underestimate the true one by a tiny bit sometimes. That's expected?
    //
    // We need a better way to *grind* these tests to verify if the confidence levels hold! That is
    // determine the variance of the test itself.
    let e95_constraint = ConfidenceLevel::P95.apply_constraint_maximizer(v, n);
    let sure_constraint = ConfidenceLevel::from_magnitude(8.5).apply_constraint_maximizer(v, n);
    (
        e95_constraint.estimate.hc_squared,
        sure_constraint.estimate.hc_squared,
    )
}

fn normal_square_bc(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (m1, s1) = a;
    let (m2, s2) = b;

    let v1 = s1 * s1;
    let v2 = s2 * s2;

    ((2.0 * s1 * s2) / (v1 + v2)).sqrt() * (-0.25 * (m1 - m2).powi(2) / (v1 + v2)).exp()
}
