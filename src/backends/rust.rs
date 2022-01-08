pub struct Rust;
use super::*;

// TODO: This needs to check if SIMD is available at compile time.
macro_rules! impl_slas_dot {
    ($t: ty, $fn: ident) => {
        /// Pure rust implementation of dot product. This is more performant for smaller vectors, where as the blas ([`cblas_sdot`] and [`cblas_ddot`]) dot product is faster for larger vectors.
        ///
        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        /// assert!(slas_sdot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
        /// ```
        fn $fn<const LEN: usize>(
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
    };
}

impl operations::DotProduct for Rust {
    impl_slas_dot!(f32, sdot);
    impl_slas_dot!(f64, ddot);
}
