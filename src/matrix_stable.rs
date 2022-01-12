use crate::{prelude::*, StaticVecUnion};
use std::marker::{PhantomData, PhantomPinned};
use std::mem::transmute_copy;
use std::ops::*;

///A stable matrix type not backed by a tensor.
///`Matrix` uses `StaticCowVec` internally, so it also has copy-on-write behaviour.
///
/// A matrix of type `Matrix<f32, M, K>` has `M` rows and `K` columns.
/// This means the height of the matrix (y len) is specified first, and then the width (x len).
///
/// When indexing into a matrix, it is in the order `[x, y]`.
///
///## Example
///```rust
/// use slas::prelude::*;
/// use slas::matrix::Matrix;
///
/// let m: Matrix<f32, 2, 3> = [
///  1., 2., 3.,
///  4., 5., 6.
/// ].moo_owned().matrix();
///
/// assert!(m[[1, 0]] == 2.);
///
/// let k = Matrix::<f32, 3, 2>(moo![f32: 0..6].moo_owned());
///
/// println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
///```
///
///I found that [Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:properties-of-matrix-multiplication/a/matrix-multiplication-dimensions)
///was a good resource for better understanding matricies.
#[derive(Copy, Clone)]
pub struct Matrix<'a, T: Copy + 'a, const M: usize, const K: usize>(pub *mut T, pub &'a ());

impl<'a, T: Copy, const M: usize, const K: usize> Matrix<'a, T, M, K>
where
    T: Float,
    StaticVecUnion<'a, T, { M * K }>: Sized,
    [T; M * K]: Sized,
{
    pub fn moo(&self) -> StaticCowVec<'a, T, { M * K }> {
        unsafe { StaticCowVec::from_ptr(self.0) }
    }
    pub fn moo_ref(&self) -> &'a StaticVecUnion<'a, T, { M * K }> {
        use std::mem::transmute;
        unsafe { transmute(self.0) }
    }
    pub fn mut_moo_ref(&self) -> &'a mut StaticVecUnion<'a, T, { M * K }> {
        use std::mem::transmute;
        unsafe { transmute(self.0) }
    }

    pub unsafe fn get_unchecked_mut(&mut self, n: [usize; 2]) -> &mut T {
        self.mut_moo_ref().get_unchecked_mut(n[0] + n[1] * K)
    }
    pub unsafe fn get_unchecked(&self, n: [usize; 2]) -> &T {
        self.mut_moo_ref().get_unchecked(n[0] + n[1] * K)
    }

    /// Very slow, quick and dirty matrix transpose. It just switches the x and y axis...
    pub fn transpose(&self) -> Matrix<T, K, M>
    where
        StaticCowVec<'a, T, { M * K }>: Sized,
    {
        let mut buffer = [0f32; M * K];
        use std::mem::transmute;
        let mut buffer = unsafe { Matrix(transmute(&mut buffer), &()) };

        for x in 0..K {
            for y in 0..M {
                unsafe { *buffer.get_unchecked_mut([y, x]) = *self.get_unchecked([x, y]) }
            }
        }
        buffer
    }
}

impl<'a, T: Copy, const M: usize, const K: usize> Deref for Matrix<'a, T, M, K>
where
    StaticVecUnion<'a, T, { K * M }>: Sized,
{
    type Target = StaticVecUnion<'a, T, { K * M }>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: Copy, const M: usize, const K: usize> DerefMut for Matrix<'a, T, M, K>
where
    StaticVecUnion<'a, T, { K * M }>: Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: Copy, const M: usize, const K: usize> Index<[usize; 2]> for Matrix<'a, T, M, K>
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

impl<'a, T: Copy, const M: usize, const K: usize> IndexMut<[usize; 2]> for Matrix<'a, T, M, K>
where
    StaticVecUnion<'a, T, { K * M }>: Sized,
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

impl<'a, T: Copy, const M: usize, const K: usize> From<StaticVecUnion<'a, T, { K * M }>>
    for Matrix<'a, T, M, K>
{
    fn from(v: StaticVecUnion<'a, T, { K * M }>) -> Self {
        Matrix(v)
    }
}

impl<'a, T: Copy, const M: usize, const K: usize> From<&'a [T; K * M]> for Matrix<'a, T, M, K> {
    fn from(v: &'a [T; K * M]) -> Self {
        unsafe { Matrix(transmute_copy(v)) }
    }
}

impl<'a, T: Copy, const M: usize, const K: usize> From<[T; K * M]> for Matrix<'a, T, M, K> {
    fn from(v: [T; K * M]) -> Self {
        unsafe { Matrix(transmute_copy(&v)) }
    }
}

impl<'a, T: Copy + std::fmt::Debug, const M: usize, const K: usize> std::fmt::Debug
    for Matrix<'a, T, M, K>
where
    StaticCowVec<'a, T, { K * M }>: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        /// This is matrix multiplication, **NOT** element wise multiplication.
        /// Take a look at
        /// [wiki](https://en.wikipedia.org/wiki/Matrix_multiplication),
        /// [3Blue1Brown at yt](https://www.youtube.com/watch?v=XkY2DOUCWMU)
        /// and/or [Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:properties-of-matrix-multiplication/a/matrix-multiplication-dimensions)
        /// for more information.
        ///
        /// It's notable that your left hand matrix needs to be as wide as the right hand matrix is tall.
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
    pub use super::*;
}
