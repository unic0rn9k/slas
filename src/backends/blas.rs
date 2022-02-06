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
                A: StaticVec<$t, ALEN>,
                B: StaticVec<$t, BLEN>,
                C: StaticVec<$t, CLEN>,
                const ALEN: usize,
                const BLEN: usize,
                const CLEN: usize,
            >(
                &self,
                a: &A,
                b: &B,
                buffer: &mut C,
                m: usize,
                n: usize,
                k: usize,
                a_trans: bool,
                b_trans: bool,
            ) where
                A: Sized,
                B: Sized,
            {
                use cblas_sys::CBLAS_TRANSPOSE::*;
                unsafe {
                    // TODO: gemv should be used here when other's dimensions are a transpose of self.
                    cblas_sys::$f(
                        cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                        if a_trans { CblasTrans } else { CblasNoTrans },
                        if b_trans { CblasTrans } else { CblasNoTrans },
                        m as i32,
                        n as i32,
                        k as i32,
                        1.,
                        a.as_ptr(),
                        k as i32,
                        b.as_ptr(),
                        n as i32,
                        0.,
                        buffer.as_ptr() as *mut $t,
                        n as i32,
                    )
                }
            }
        }
    };
}

macro_rules! impl_norm {
    ($t: ty, $t2: ty, $t3: ty, $blas_fn: ident) => {
        impl operations::Normalize<$t> for Blas {
            type NormOutput = $t3;
            fn norm<const LEN: usize>(&self, a: &impl StaticVec<$t, LEN>) -> $t3 {
                unsafe { cblas_sys::$blas_fn(LEN as i32, a.as_ptr() as *const $t2, 1) }.into()
            }

            fn normalize<const LEN: usize>(&self, a: &mut impl StaticVec<$t, LEN>) {
                let norm = <$t>::from(Backend::norm(self, a));
                a.mut_moo_ref().iter_mut().for_each(|n| *n = *n / norm);
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

impl_norm!(f32, f32, f32, cblas_snrm2);
impl_norm!(f64, f64, f64, cblas_dnrm2);
impl_norm!(Complex<f32>, [f32; 2], f32, cblas_scnrm2);
impl_norm!(Complex<f64>, [f64; 2], f64, cblas_dznrm2);

impl Backend<f32> for Blas {}
impl Backend<f64> for Blas {}
impl Backend<Complex<f32>> for Blas {}
impl Backend<Complex<f64>> for Blas {}
