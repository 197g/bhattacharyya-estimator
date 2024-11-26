use rand::{distributions::OpenClosed01, Rng as _};
use statrs::distribution::{ContinuousCDF as _, Normal};

#[test]
fn generate_sample() {
    let n = Normal::new(0.0, 1.0).unwrap();
    const COUNT: usize = 1_000_000;

    let v: Vec<f64> = rand::thread_rng()
        .sample_iter(OpenClosed01)
        .take(COUNT)
        .map(|x| n.inverse_cdf(x))
        .collect();
}
