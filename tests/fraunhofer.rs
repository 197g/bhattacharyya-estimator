//! Can we detect Fraunhofer lines?
//!
//! The sun is speculated to be a 5777K black body. We have a spectrum. Can those two distributions
//! be reliably differentiated from each other? Recall Planck's law.
//!
//! ```
//! radiance(lambda) = (2hc²)/(lamda^5) / (exp(hc/lambda·k·T) - 1)
//! ```

use estimated_hellinger::Estimate;

#[test]
fn fraunhofer() {
    let (nm, mut real) = parse_wehrli();
    let mut theory = black_body(&nm, 5777.0);

    to_cdf(&mut theory);
    to_cdf(&mut real);

    // Title: Improved data of solar spectral irradiance from 0.33 to 1.25 microns
    // Authors: Neckel, H. & Labs, D.
    //
    // Puts the error bounds on 0.6% for each individual measurement. Let's be a little arrogant
    // and suggest that this is noise and the CDF we got is actually better?
    let test = Estimate::from_matched_cdf(&real, &theory, 0.006);
    assert!(test.bc_estimate > 1.0, "Inconclusive from too little data");
}

#[test]
fn astm_2000() {
    let (mut nm, mut real) = parse_astm_2000();
    nm.iter_mut().for_each(|x| *x *= 1000.);
    let mut theory = black_body(&nm, 5777.0);

    to_cdf(&mut theory);
    to_cdf(&mut real);

    let ahaha = Estimate::from_matched_cdf(&real, &theory, 0.);
    assert!(ahaha.bc_estimate < 1.0, "Conclusive ???");
}

fn to_cdf(vec: &mut Vec<f64>) {
    let sum: f64 = vec.iter_mut().fold(0.0, |mut sum, x| {
        sum += *x;
        *x = sum;
        sum
    });

    vec.iter_mut().for_each(|x| *x /= sum);
}

fn parse_wehrli() -> (Vec<f64>, Vec<f64>) {
    // From: <https://www.nrel.gov/grid/solar-resource/spectra-wehrli.html>
    const WERHLI: &str = include_str!("./wehrli85.txt");
    parse_observation(WERHLI)
}

fn parse_astm_2000() -> (Vec<f64>, Vec<f64>) {
    // From: <https://www.nrel.gov/grid/solar-resource/spectra-wehrli.html>
    const ASTM: &str = include_str!("./astm-2000-standard-spectrum-e490.csv");
    parse_observation(ASTM)
}

fn parse_observation(st: &str) -> (Vec<f64>, Vec<f64>) {
    let mut nm = vec![];
    let mut irr = vec![];

    for line in st.lines().skip(2) {
        let mut line = line.split_ascii_whitespace();
        let txt_nm = line.next().unwrap();
        let txt_irr = line.next().unwrap();

        nm.push(txt_nm.parse().unwrap());
        irr.push(txt_irr.parse().unwrap());
    }

    (nm, irr)
}

fn black_body(nm: &[f64], t: f64) -> Vec<f64> {
    const CONST_H: f64 = 6.62607015e-34;
    const CONST_C: f64 = 299792458.;
    const CONST_K: f64 = 1.380649e-23;

    nm.iter()
        .copied()
        .map(|l| {
            let l = l * 1.0e-9;
            let num = (2.0 * CONST_H * CONST_C * CONST_C) / l.powf(5.0);
            let denom = ((CONST_H * CONST_C) / (l * CONST_K * t)).exp() - 1.0;
            num / denom
        })
        .collect()
}
