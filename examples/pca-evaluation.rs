use nalgebra::{Dyn, Matrix};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use ndarray::Array2;
use single_algebra::dimred::pca;

use hellinger_estimates::ConfidenceLevel;

fn main() {
    let data = load_to_rows();
    let pca = to_pca(&data, 8);

    for (idx, col) in pca.columns().into_iter().enumerate() {
        let level = ConfidenceLevel::P99.non_normality_constraint(col.iter().copied());
        println!(
            "Dir {idx}: {:.6} from normal",
            level.estimate.hc_squared.max(0.0)
        );
    }
}

fn to_pca(data: &Array2<f64>, n_components: usize) -> Array2<f64> {
    let (nrows, ncols) = data.dim();

    let data = data.iter().copied().collect::<Vec<_>>();
    let data = Matrix::<f64, Dyn, Dyn, _>::from_row_slice(nrows, ncols, &data);

    let mut coo = CooMatrix::new(nrows, ncols);
    coo.push_matrix(0, 0, &data);
    let csr = CsrMatrix::from(&coo);

    let mut pca = pca::SparsePCA::<f64>::new(
        n_components,
        0.0,
        None,
        None,
        false, /* center */
        false, /* verbose */
        pca::SVDMethod::Lanczos,
    );

    pca.fit_transform(&csr).unwrap()
}

fn load_to_rows() -> Array2<f64> {
    linfa_datasets::winequality().records
}
