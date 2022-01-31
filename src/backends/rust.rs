/// A pure rust slas backend with simd support.
#[derive(Default)]
pub struct Rust;
use super::*;
use std::simd::Simd;

// TODO: This needs to check if SIMD is available at compile time.
macro_rules! impl_dot {
    ($t: ty) => {
        /// Pure rust implementation of dot product. This is more performant for smaller vectors,
        /// where as the blas (cblas_sdot and cblas_ddot) dot products are faster for larger vectors.
        ///
        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        /// assert!(slas_backend::Rust.dot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
        /// ```
        impl operations::DotProduct<$t> for Rust {
            fn dot<const LEN: usize>(
                &self,
                a: &impl StaticVec<$t, LEN>,
                b: &impl StaticVec<$t, LEN>,
            ) -> $t {
                const LANES: usize = 4;
                let mut sum = Simd::<$t, LANES>::from_array([0.; LANES]);
                for n in 0..LEN / LANES {
                    sum += unsafe {
                        Simd::from_slice(a.static_slice_unchecked::<LANES>(n * LANES))
                            * Simd::from_slice(b.static_slice_unchecked::<LANES>(n * LANES))
                    }
                }
                let mut sum = sum.horizontal_sum();
                for n in LEN - (LEN % LANES)..LEN {
                    sum += unsafe { a.get_unchecked(n) * b.get_unchecked(n) }
                }
                sum
            }
        }
    };
}

macro_rules! impl_norm {
    ($t: ty) => {
        impl operations::Normalize<$t> for Rust {
            type NormOutput = $t;
            fn norm<const LEN: usize>(&self, a: &impl StaticVec<$t, LEN>) -> $t {
                //TODO: Use hypot function here. This will require implementing hypot for all float types first.
                a.moo_ref().iter().map(|&n| n * n).sum::<$t>().sqrt_()
            }

            fn normalize<const LEN: usize>(&self, a: &mut impl StaticVec<$t, LEN>) {
                let norm = operations::Normalize::norm(self, a);
                a.mut_moo_ref().iter_mut().for_each(|n| *n = *n / norm);
            }
        }

        impl operations::Normalize<Complex<$t>> for Rust {
            type NormOutput = $t;
            fn norm<const LEN: usize>(&self, a: &impl StaticVec<Complex<$t>, LEN>) -> $t {
                //TODO: Use hypot function here. This will require implementing hypot for all float types first.
                a.moo_ref()
                    .iter()
                    .map(|n| Simd::<$t, 2>::from_array([n.re.powi_(2), n.im.powi_(2)]))
                    .sum::<Simd<$t, 2>>()
                    .horizontal_sum()
                    .sqrt()
            }

            fn normalize<const LEN: usize>(&self, a: &mut impl StaticVec<Complex<$t>, LEN>) {
                let norm = operations::Normalize::norm(self, a);
                a.mut_moo_ref()
                    .iter_mut()
                    .for_each(|n| *n = *n / norm.into());
            }
        }
    };
}

impl_norm!(f32);
impl_norm!(f64);

impl_dot!(f32);
impl_dot!(f64);

impl Backend<f32> for Rust {}
impl Backend<f64> for Rust {}
impl Backend<Complex<f32>> for Rust {}
impl Backend<Complex<f64>> for Rust {}
