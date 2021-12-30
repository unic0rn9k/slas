use crate::prelude::*;
use num::NumCast;
use std::ops::*;

#[derive(Copy, Clone)]
pub struct Matrix<'a, T: NumCast + Copy, const M: usize, const K: usize>(
    StaticCowVec<'a, T, { K * M }>,
)
where
    StaticCowVec<'a, T, { K * M }>: Sized;

impl<'a, T: NumCast + Copy, const M: usize, const K: usize> Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    pub fn zeros() -> Self {
        Self(StaticCowVec::zeros())
    }

    pub fn is_borrowed(&self) -> bool {
        self.0.is_borrowed()
    }

    pub fn is_owned(&self) -> bool {
        self.0.is_owned()
    }

    pub unsafe fn get_unchecked_mut(&mut self, n: [usize; 2]) -> &mut T {
        self.0.get_unchecked_mut(n[0] + n[1] * K)
    }
    pub unsafe fn get_unchecked(&self, n: [usize; 2]) -> &T {
        self.0.get_unchecked(n[0] + n[1] * K)
    }

    pub fn transpose(&self) -> Matrix<T, K, M>
    where
        StaticCowVec<'a, T, { M * K }>: Sized,
    {
        let mut buffer = Matrix::<T, K, M>::zeros();
        for x in 0..K {
            for y in 0..M {
                unsafe { *buffer.get_unchecked_mut([y, x]) = *self.get_unchecked([x, y]) }
            }
        }
        buffer
    }
}

impl<'a, T: NumCast + Copy, const M: usize, const K: usize> Deref for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    type Target = StaticCowVec<'a, T, { K * M }>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: NumCast + Copy, const M: usize, const K: usize> DerefMut for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: NumCast + Copy, const M: usize, const K: usize> Index<[usize; 2]>
    for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    type Output = T;
    fn index(&self, n: [usize; 2]) -> &T {
        assert!(
            n[0] < K && n[1] < M,
            "Index {:?} out of bounds {:?}",
            n,
            [K, M]
        );
        unsafe { self.0.get_unchecked(n[0] + n[1] * K) }
    }
}

impl<'a, T: NumCast + Copy, const M: usize, const K: usize> IndexMut<[usize; 2]>
    for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    fn index_mut(&mut self, n: [usize; 2]) -> &mut T {
        assert!(
            n[0] < K && n[1] < M,
            "Index {:?} out of bounds {:?}",
            n,
            [K, M]
        );
        unsafe { self.0.get_unchecked_mut(n[0] + n[1] * K) }
    }
}

impl<'a, T: Copy + NumCast, const M: usize, const K: usize> From<StaticCowVec<'a, T, { K * M }>>
    for Matrix<'a, T, M, K>
{
    fn from(v: StaticCowVec<'a, T, { K * M }>) -> Self {
        Matrix(v)
    }
}

impl<'a, T: Copy + NumCast, const M: usize, const K: usize> From<&'a [T; K * M]>
    for Matrix<'a, T, M, K>
{
    fn from(v: &'a [T; K * M]) -> Self {
        Matrix(v.into())
    }
}

impl<'a, T: Copy + NumCast, const M: usize, const K: usize> From<[T; K * M]>
    for Matrix<'a, T, M, K>
{
    fn from(v: [T; K * M]) -> Self {
        Matrix(v.into())
    }
}

impl<'a, T: Copy + NumCast + std::fmt::Debug, const M: usize, const K: usize> std::fmt::Debug
    for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        if self.is_borrowed() {
            f.write_char('&')?;
        }
        f.write_str("[\n")?;
        for n in 0..M {
            f.write_str("   ")?;
            f.debug_list()
                .entries(self.0[n * K..(n + 1) * K].iter())
                .finish()?;
            f.write_str(",\n")?;
        }
        f.write_str("]")?;
        Ok(())
    }
}

macro_rules! impl_gemm {
    ($t: ty, $f: ident) => {
        impl<'a, 'b, const M: usize, const N: usize, const K: usize> Mul<Matrix<'b, $t, K, N>>
            for Matrix<'a, $t, M, K>
        where
            StaticCowVec<'a, $t, { K * M }>: Sized,
            StaticCowVec<'a, $t, { N * K }>: Sized,
            StaticCowVec<'a, $t, { N * M }>: Sized,
        {
            type Output = Matrix<'static, $t, M, N>;
            fn mul(self, other: Matrix<'b, $t, K, N>) -> Self::Output {
                let mut buffer = Self::Output::zeros();
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
                        self.as_ptr(),
                        K as i32,
                        other.as_ptr(),
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

pub mod matrix {
    pub use super::Matrix;
}
