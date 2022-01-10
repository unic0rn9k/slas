#[derive(Default)]
pub struct Blas;
use super::*;

macro_rules! impl_dot {
    ($t: ty, $blas_fn: ident) => {
        /// Thin wrapper around blas for the various dot product functions that works for multiple different (and mixed) vector types.
        ///
        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        /// assert!(slas_backend::Blas.sdot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
        /// ```
        impl operations::DotProduct<$t> for Blas {
            fn dot<const LEN: usize>(
                &self,
                a: &impl StaticVec<$t, LEN>,
                b: &impl StaticVec<$t, LEN>,
            ) -> $t {
                unsafe { cblas_sys::$blas_fn(LEN as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
            }
        }
    };
}

macro_rules! impl_dot_comp {
    ($t: ty, $comp_blas_fn: ident) => {
        /// Dot product for two complex vectors.
        /// Also has support for multiple (and mixed) types.
        impl operations::DotProduct<Complex<$t>> for Blas {
            fn dot<const LEN: usize>(
                &self,
                a: &impl StaticVec<Complex<$t>, LEN>,
                b: &impl StaticVec<Complex<$t>, LEN>,
            ) -> Complex<$t> {
                let mut tmp: [$t; 2] = [0.; 2];
                unsafe {
                    cblas_sys::$comp_blas_fn(
                        LEN as i32,
                        a.as_ptr() as *const [$t; 2],
                        1,
                        b.as_ptr() as *const [$t; 2],
                        1,
                        tmp.as_mut_ptr() as *mut [$t; 2],
                    )
                }
                Complex {
                    re: tmp[0],
                    im: tmp[1],
                }
            }
        }
    };
}

impl_dot!(f32, cblas_sdot);
impl_dot!(f64, cblas_ddot);
impl_dot_comp!(f32, cblas_cdotu_sub);
impl_dot_comp!(f64, cblas_zdotu_sub);
impl Backend<f32> for Blas {}
impl Backend<f64> for Blas {}
impl Backend<Complex<f32>> for Blas {}
impl Backend<Complex<f64>> for Blas {}
