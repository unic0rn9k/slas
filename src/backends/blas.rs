/// Binding to blas with [cblas-sys](https://lib.rs/cblas-sys) as a slas backend.
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
        /// assert!(slas_backend::Blas.dot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
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

macro_rules! impl_gemm {
    ($t: ty, $f: ident) => {
        /// This is matrix multiplication, **NOT** element wise multiplication.
        /// Take a look at
        /// [wiki](https://en.wikipedia.org/wiki/Matrix_multiplication),
        /// [3Blue1Brown at yt](https://www.youtube.com/watch?v=XkY2DOUCWMU)
        /// and/or [Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:properties-of-matrix-multiplication/a/matrix-multiplication-dimensions)
        /// for more information.
        ///
        /// It's notable that your left hand matrix needs to be as wide as the right hand matrix is tall.
        impl operations::MatrixMul<$t> for Blas {
            fn matrix_mul<
                A: StaticVec<$t, { M * K }>,
                B: StaticVec<$t, { N * K }>,
                const M: usize,
                const N: usize,
                const K: usize,
            >(
                &self,
                a: &A,
                b: &B,
            ) -> [$t; N * M]
            where
                A: Sized,
                B: Sized,
                [$t; N * M]: Sized,
            {
                let mut buffer = [<$t>::zero(); N * M];
                unsafe {
                    // TODO: gemv should be used here when other's dimensions are a transpose of self.
                    cblas_sys::$f(
                        cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        M as i32,
                        N as i32,
                        K as i32,
                        1.,
                        a.as_ptr(),
                        K as i32,
                        b.as_ptr(),
                        N as i32,
                        0.,
                        buffer.as_mut_ptr(),
                        N as i32,
                    )
                }
                buffer
            }
        }
    };
}

impl_gemm!(f32, cblas_sgemm);
impl_gemm!(f64, cblas_dgemm);

impl_dot!(f32, cblas_sdot);
impl_dot!(f64, cblas_ddot);
impl_dot_comp!(f32, cblas_cdotu_sub);
impl_dot_comp!(f64, cblas_zdotu_sub);
impl Backend<f32> for Blas {}
impl Backend<f64> for Blas {}
impl Backend<Complex<f32>> for Blas {}
impl Backend<Complex<f64>> for Blas {}
