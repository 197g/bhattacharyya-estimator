use statrs::distribution::ContinuousCDF;

#[derive(Clone, Debug)]
pub struct Estimate {
    /// An estimate of the Bhattacharyya coefficient.
    ///
    /// On average this is an over estimation. In the limit of samples it is the true coefficient.
    pub bc_estimate: f64,
    pub hc_squared: f64,
    pub total_variance_upper: f64,
}

pub struct ConfidenceLevel {
    dkw_constant: f64,
}

impl ConfidenceLevel {
    /// Special confidence level that does not provide any confidence.
    pub const NONE: Self = ConfidenceLevel { dkw_constant: 0.0 };

    /// Pre-computed confidence level with less than 5% error.
    pub const P95: Self = ConfidenceLevel {
        dkw_constant: 1.3581015157406195,
    };

    /// Pre-computed confidence level with less than 2% error.
    pub const P98: Self = ConfidenceLevel {
        dkw_constant: 1.5174271293851462,
    };

    /// Pre-computed confidence level with less than 1% error.
    pub const P99: Self = ConfidenceLevel {
        dkw_constant: 1.6276236307187293,
    };

    pub fn new(level: f64) -> Self {
        assert!(level > 0.0);
        assert!(level < 1.0);

        ConfidenceLevel {
            dkw_constant: Self::dvoretzky_kiefer_wolfowitz_constant(level),
        }
    }

    pub fn apply(&self, sorted: &[f64], cdf: &dyn ContinuousCDF<f64, f64>) -> Estimate {
        assert!(!sorted.is_empty(), "No estimate for empty sample");

        let count = sorted.len() as f64;
        let sqrt_n = count.sqrt();

        let skip = count.powf(2.0 / 3.0).ceil() as usize;

        // Quantiles according to limit distribution.
        let ps = (0..sorted.len())
            .step_by(skip)
            .map(|n| (1. + n as f64) / count);

        // Quantiles according to model CDF.
        let qs = sorted.chunks(skip).map(|arr| cdf.cdf(arr[0]));

        let expand = self.dkw_constant / sqrt_n;
        Estimate::from_matched_quantiles(ps.collect(), qs.collect(), expand)
    }

    /// Adjustment of CDF with Dvoretzky-Kiefer-Wolfowitz bounds:
    ///
    /// P(sup|F_n - F| > eps) <= 2 * exp(-2n * eps**2)
    ///
    /// We choose the confidence level `P(..) <= 2 * exp(-2n * eps**2) <= p`.
    ///
    /// ```text
    /// 1 >= (2/p) * exp(-2n * eps**2)
    /// 0 >= log(2/p) + (-2n * eps**2)
    /// eps**2 >= log(2/p) / (2n)
    /// eps >= sqrt(log(2 / p) / 2) / sqrt(n)
    /// ```
    ///
    /// This only returns the constant which does not depend on `n`.
    fn dvoretzky_kiefer_wolfowitz_constant(level: f64) -> f64 {
        assert!(level < 1.0);
        ((2.0f64.ln() - level.ln()) / 2.0).sqrt()
    }
}

impl Estimate {
    pub fn from_ecdf(sorted: &[f64], cdf: &dyn ContinuousCDF<f64, f64>) -> Self {
        assert!(!sorted.is_empty(), "No estimate for empty sample");

        let count = sorted.len() as f64;
        let skip = count.sqrt().ceil() as usize;

        // Quantiles according to limit distribution.
        let ps = (0..sorted.len())
            .step_by(skip)
            .map(|n| (1. + n as f64) / count);

        // Quantiles according to model CDF.
        let qs = sorted.chunks(skip).map(|arr| cdf.cdf(arr[0]));

        Self::from_matched_quantiles(ps.collect(), qs.collect(), 0.0)
    }

    fn from_matched_quantiles(mut ps: Vec<f64>, mut qs: Vec<f64>, expand: f64) -> Self {
        Self::diff_in_place_with_added_bias(&mut ps, 0.0);
        Self::diff_in_place_with_added_bias(&mut qs, expand);

        let p_weight = ps.iter().copied();
        let q_weight = qs.iter().copied();

        // Over-estimation of the real, plus bias.
        let bc_estimate: f64 = p_weight
            .zip(q_weight)
            .map(|(lp, lq)| (lp * lq).sqrt())
            .sum();

        // Under-estimation of H²
        let hc_squared = 1.0 - bc_estimate;
        // High Bound of TVD based on fundamental form of Hellinger distance.
        let total_variance_upper = 2.0f64.sqrt() * hc_squared.sqrt();

        Estimate {
            bc_estimate,
            hc_squared,
            total_variance_upper,
        }
    }

    fn diff_in_place_with_added_bias(slice: &mut [f64], expand: f64) {
        let mut state = 0.0;
        slice.iter_mut().for_each(|x| {
            let pre = core::mem::replace(&mut state, *x);
            *x = (*x + expand).clamp(0.0, 1.0) - (pre - expand).clamp(0.0, 1.0);
        })
    }
}
