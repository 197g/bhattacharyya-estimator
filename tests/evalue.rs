use estimated_hellinger::MaximumHellingerHypothesis;

use rand::{distributions::OpenClosed01, Rng as _};
use statrs::distribution::{ContinuousCDF, Normal};

#[test]
fn test_evalue() {
    let n = Normal::new(0.0, 1.0).unwrap();
    const COUNT: usize = 1_000_000;

    let mut v: Vec<f64> = rand::thread_rng()
        .sample_iter(OpenClosed01)
        .take(COUNT)
        .map(|x| n.inverse_cdf(x))
        .collect();

    v.sort_by(|a, b| a.total_cmp(b));

    for (mean, stddev) in [
        (0.0, 1.0),
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
    fn verify(actual: f64, hypoth: f64, evalue: estimated_hellinger::Evalue) -> &'static str {
        // E-Value requires that the expectation <= 1.0 in the world where hypothesis holds.
        // So under-reporting is simply a failure to reject, over reporting a problem with the
        // E-value. Of course, the setup below only does a single sample and not in fact a real
        // verification of the expected value (e.g a t-test against samples of computed E-Values).
        // Please fix this by contribution if you read this.
        if actual > hypoth {
            if evalue.value >= 1.0 {
                "PASS"
            } else {
                // Okay, but suboptimal.
                "NO REJECTION"
            }
        } else {
            if evalue.value <= 1.0 {
                "PASS"
            } else {
                "FAIL"
            }
        }
    }

    // The bad confidence levels, and in particular none-confidence level, will happen to
    // underestimate the true one by a tiny bit sometimes. That's expected?
    //
    // We need a better way to *grind* these tests to verify if the confidence levels hold! That is
    // determine the variance of the test itself.
    let hellinger = (1.0 - actual_bc).sqrt();
    eprintln!("Actual {hellinger}");

    let evalue = MaximumHellingerHypothesis::new(hellinger).e_value(v, n);
    eprintln!("E-Value ({hellinger:.4}) {} {}", evalue.value, verify(hellinger, hellinger, evalue));

    let evalue = MaximumHellingerHypothesis::new(1.0 / 3.0).e_value(v, n);
    eprintln!("E-Value (0.3333) {} {}", evalue.value, verify(hellinger, 1.0 / 3.0, evalue));

    let strictish = (hellinger.ln() - 0.25).exp();
    let evalue = MaximumHellingerHypothesis::new(strictish).e_value(v, n);
    eprintln!("E-Value ({strictish:.4}) {} {}", evalue.value, verify(hellinger, strictish, evalue));

    let strictish = (hellinger.ln() - 0.0625).exp();
    let evalue = MaximumHellingerHypothesis::new(strictish).e_value(v, n);
    eprintln!("E-Value ({strictish:.4}) {} {}", evalue.value, verify(hellinger, strictish, evalue));

    let strictish = (hellinger.ln() - 0.01575).exp();
    let evalue = MaximumHellingerHypothesis::new(strictish).e_value(v, n);
    eprintln!("E-Value ({strictish:.4}) {} {}", evalue.value, verify(hellinger, strictish, evalue));
}

fn normal_square_bc(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (m1, s1) = a;
    let (m2, s2) = b;

    let v1 = s1 * s1;
    let v2 = s2 * s2;

    ((2.0 * s1 * s2) / (v1 + v2)).sqrt() * (-0.25 * (m1 - m2).powi(2) / (v1 + v2)).exp()
}
