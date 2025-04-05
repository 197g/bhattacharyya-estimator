use estimated_hellinger::{mixed, ConfidenceLevel};

use rand::Rng as _;
use statrs::distribution::{ContinuousCDF, Normal};

struct Mixture2 {
    lambda: f64,
    left: Normal,
    right: Normal,
    h2_precise: f64,
}

#[test]
fn generate_sample() {
    fn compare_sample(a: (f64, f64), b: (f64, f64), lambda: f64) {
        let mut mixture = Mixture2::new(a, b);
        mixture.lambda = lambda;

        let mut samples = (0..1_000_000)
            .map(|_| rand::thread_rng().sample(&mixture))
            .collect::<Vec<_>>();

        samples.sort_by(|a, b| a.total_cmp(b));

        assert!(samples.iter().all(|&x| x.is_finite()));

        let mut estimator = mixed::SimplexEstimator::new(
            const { core::num::NonZeroUsize::new(2).unwrap() },
            ConfidenceLevel::from_magnitude(2.0),
        );

        mixture.add_to(&mut estimator);
        let estimate = estimator.estimate(&samples);

        for facet in &estimate.facets {
            eprintln!("{:?} / {:?}", facet.hc_squared, mixture.h2_precise);

            eprintln!("{:?}", estimate.base_of(facet));
            eprintln!("{:?}", estimate.offsets_of(facet));

            let only_facet = estimate.offsets_of(facet).next().unwrap();

            assert!(
                (estimate.base_of(facet)[1] < mixture.lambda) == (only_facet[1] > 0.0),
                "The facet must point towards the mix lambda"
            );
        }
    }

    compare_sample((-2.0, 1.0), (2.0, 1.0), 0.4);
    compare_sample((-2.0, 1.0), (2.0, 1.0), 0.1);
    compare_sample((-2.0, 1.0), (2.0, 1.0), 0.9);

    compare_sample((0.0, 1.0), (0.0, 2.0), 0.5);
}

impl statrs::statistics::Min<f64> for Mixture2 {
    fn min(&self) -> f64 {
        -f64::INFINITY
    }
}

impl statrs::statistics::Max<f64> for Mixture2 {
    fn max(&self) -> f64 {
        -f64::INFINITY
    }
}

impl ContinuousCDF<f64, f64> for Mixture2 {
    fn cdf(&self, x: f64) -> f64 {
        self.lambda * self.left.cdf(x) + (1.0 - self.lambda) * self.right.cdf(x)
    }
}

impl rand::distributions::Distribution<f64> for Mixture2 {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if rng.gen::<f64>() < self.lambda {
            self.left.sample(rng)
        } else {
            self.right.sample(rng)
        }
    }
}

impl Mixture2 {
    fn new(a: (f64, f64), b: (f64, f64)) -> Self {
        Mixture2 {
            lambda: 0.5,
            left: Normal::new(a.0, a.1).unwrap(),
            right: Normal::new(b.0, b.1).unwrap(),
            h2_precise: 1. - Self::normal_square_bc(a, b),
        }
    }

    fn add_to(&self, simple: &mut mixed::SimplexEstimator) {
        simple.add_distribution(self.left.clone(), &[0.0, 1.0, 1.0, 0.0], &[self.h2_precise]);
        simple.add_distribution(
            self.right.clone(),
            &[1.0, 0.0, 0.0, 1.0],
            &[self.h2_precise],
        );
    }

    fn normal_square_bc(a: (f64, f64), b: (f64, f64)) -> f64 {
        let (m1, s1) = a;
        let (m2, s2) = b;

        let v1 = s1 * s1;
        let v2 = s2 * s2;

        ((2.0 * s1 * s2) / (v1 + v2)).sqrt() * (-0.25 * (m1 - m2).powi(2) / (v1 + v2)).exp()
    }
}
