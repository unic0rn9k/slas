pub struct Blas;
use super::*;

macro_rules! impl_dot {
    ($float: ty, $blas_fn: ident, $slas_fn: ident) => {
        /// Thin wrapper around blas for the various dot product functions that works for multiple different (and mixed) vector types.
        ///
        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        /// assert!(cblas_sdot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
        /// ```
        fn $slas_fn<const LEN: usize>(
            &self,
            a: &impl StaticVec<$float, LEN>,
            b: &impl StaticVec<$float, LEN>,
        ) -> $float {
            unsafe { cblas_sys::$blas_fn(LEN as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
        }
    };
}

macro_rules! impl_dot_comp {
    ($float: ty, $comp_blas_fn: ident, $slas_fn: ident) => {
        /// Dot product for two complex vectors.
        /// Also has support for multiple (and mixed) types.
        fn $slas_fn<const LEN: usize>(
            &self,
            a: &impl StaticVec<Complex<$float>, LEN>,
            b: &impl StaticVec<Complex<$float>, LEN>,
        ) -> Complex<$float> {
            let mut tmp: [$float; 2] = [0.; 2];
            unsafe {
                cblas_sys::$comp_blas_fn(
                    LEN as i32,
                    a.as_ptr() as *const [$float; 2],
                    1,
                    b.as_ptr() as *const [$float; 2],
                    1,
                    tmp.as_mut_ptr() as *mut [$float; 2],
                )
            }
            Complex {
                re: tmp[0],
                im: tmp[1],
            }
        }
    };
}

impl operations::DotProduct for Blas {
    impl_dot!(f32, cblas_sdot, sdot);
    impl_dot!(f64, cblas_ddot, ddot);
}
impl operations::ComplexDotProduct for Blas {
    impl_dot_comp!(f32, cblas_cdotu_sub, cdotu_sub);
    impl_dot_comp!(f64, cblas_zdotu_sub, zdotu_sub);
}
