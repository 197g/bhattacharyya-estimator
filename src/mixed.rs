//! Estimator for mixed probabilities.
//!
//! Finds bounds on the coefficients of a mix mixed probability function with high probability. You
//! give upper bounds of the squared Hellinger distance between all pairs of distributions. (For
//! instance, the TVD or any other f-divergence that also bounds H² from above can be used.) The
//! distance between empirical sample and the vertices of the MD simplex gives lower bounds on H²,
//! with high probability. Due to the joint convexity we then return facets that bound the
//! potential region.
//!
//! This probably degrades quick in higher dimensions. Also note that the upper estimates done this
//! way are not very tight outside the edges. You are free to provide some strategically chosen
//! mixed distributions as additional vertices; with their own set of distances.
//!
//! The TVD can often be analytically computed (or approximated with floating point math, please
//! ensure you errors upwards if possible). For instance for Gaussians a result is readily found
//! here: <https://arxiv.org/pdf/1810.08693>.
//!
//! Also note that the TVD itself is also upper bounded by some expressions from the
//! Kullback-Leibler divergence. See this paper which you may enjoy anyways, it's terrific:
//! <https://arxiv.org/pdf/2202.07198>

use statrs::distribution::ContinuousCDF;

use crate::ConfidenceLevel;

pub struct SimplexEstimator {
    /// The dimensionality of the simplex.
    dim: usize,
    level: ConfidenceLevel,
    vertices: Vec<Box<dyn Estimator>>,
    cones: Vec<VertexCone>,
    /// The flattened list of all point lists of cones.
    coordinates: Vec<f64>,
    /// The flattened list of all point distances.
    tvd_estimates: Vec<f64>,
}

#[non_exhaustive]
pub struct EstimateBounds {
    pub coordinates: Vec<f64>,
    pub facets: Vec<EstimatedFacet>,
    dim: usize,
}

/// A single facet.
///
/// We store the facet as follows: the first coordinate is a vertex on the outer side. Then follow
/// offset vectors to points on the facet. And of course the facet is not parallel to the simplex
/// itself.
#[non_exhaustive]
pub struct EstimatedFacet {
    pub hc_squared: f64,
    pub coordinates_start: usize,
}

struct VertexCone {
    /// The base vertex index of this cone.
    base: usize,
    coords_start: usize,
    tvd_start: usize,
}

trait Estimator {
    fn for_sample(&mut self, sample: &[f64], _: &super::ConfidenceLevel) -> super::Estimate;
}

impl SimplexEstimator {
    /// Create a new estimator.
    ///
    /// Pass the dimension of the whole space.
    pub fn new(dimension: core::num::NonZeroUsize, level: ConfidenceLevel) -> Self {
        Self {
            dim: dimension.get(),
            level,
            vertices: Vec::new(),
            cones: Vec::new(),
            coordinates: Vec::new(),
            tvd_estimates: Vec::new(),
        }
    }

    pub fn estimate(&mut self, samples: &[f64]) -> EstimateBounds {
        let mut builder = EstimateBounds {
            dim: self.dim,
            coordinates: Vec::new(),
            facets: Vec::new(),
        };

        for cone in &self.cones {
            let v = &mut self.vertices[cone.base];
            let coordinates = &self.coordinates[cone.coords_start..];
            let tvd_estimates = &self.tvd_estimates[cone.tvd_start..][..self.dim - 1];

            // Derive the lower bound on the h² from the cone base.
            let estimate = v.for_sample(samples, &self.level);
            let h2_lower = estimate.hc_squared;

            if h2_lower < 0.0 {
                // We do not have anything. At that level of confidence we can not exclude the
                // sample from actually being equivalent to the base. Just skip this one.
                continue;
            }

            // Copy everything into place.
            let coordinates_start = builder.coordinates.len();
            let (base, targets) = {
                builder.coordinates.extend_from_slice(coordinates);
                let all = &mut builder.coordinates[coordinates_start..][..self.dim * self.dim];
                all.split_at_mut(self.dim)
            };

            debug_assert_eq!(
                tvd_estimates.len() * self.dim,
                targets.len(),
                "A required when adding this cone, must have one TVD estimate for each edge."
            );

            // Determine each edges intersect with the dividing plane.
            for (target, &tvd_upper) in targets.chunks_exact_mut(self.dim).zip(tvd_estimates) {
                let scale = (h2_lower / tvd_upper).min(1.0);

                for (i, coord) in target.iter_mut().enumerate() {
                    *coord = (*coord - base[i]) * scale;
                }
            }

            builder.facets.push(EstimatedFacet {
                hc_squared: h2_lower,
                coordinates_start,
            });
        }

        builder
    }

    pub fn add_distribution(
        &mut self,
        cdf: impl ContinuousCDF<f64, f64> + 'static,
        // The coordinates of all mixes that this distribution is compared to.
        coordinates: &[f64],
        tvd_estimates: &[f64],
    ) {
        struct EstimateFromCdf<C>(C);

        impl<C: ContinuousCDF<f64, f64>> Estimator for EstimateFromCdf<C> {
            fn for_sample(
                &mut self,
                sample: &[f64],
                level: &crate::ConfidenceLevel,
            ) -> crate::Estimate {
                level.apply(sample, &self.0)
            }
        }

        let vertex = Box::new(EstimateFromCdf(cdf));
        self.add_cone(vertex, coordinates, tvd_estimates);
    }

    fn add_cone(&mut self, vertex: Box<dyn Estimator>, coordinates: &[f64], tvd_estimates: &[f64]) {
        assert!(
            coordinates.len() == self.dim * self.dim,
            "Coordinates must be of the same dimension as the simplex, expected {}",
            self.dim * self.dim,
        );

        assert!(
            tvd_estimates.len() == self.dim - 1,
            "TVD estimates must be available for each edge of the cone"
        );

        let base = self.vertices.len();
        let coords_start = self.coordinates.len();
        let tvd_start = self.tvd_estimates.len();

        self.vertices.push(vertex);
        self.coordinates.extend_from_slice(coordinates);
        self.tvd_estimates.extend_from_slice(tvd_estimates);

        self.cones.push(VertexCone {
            base,
            coords_start,
            tvd_start,
        })
    }
}

impl EstimateBounds {
    pub fn base_of(&self, facet: &EstimatedFacet) -> &[f64] {
        &self.coordinates[facet.coordinates_start..][..self.dim]
    }

    pub fn offsets_of(&self, facet: &EstimatedFacet) -> core::slice::ChunksExact<'_, f64> {
        let len = (self.dim - 1) * self.dim;
        let coords = &self.coordinates[facet.coordinates_start..];
        coords[self.dim..][..len].chunks_exact(self.dim)
    }
}
