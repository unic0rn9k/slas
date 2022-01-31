use crate::{backends::*, prelude::*};
use std::hint::unreachable_unchecked;

/// Tensor shape with static dimensions but with optionally dynamic shape.
/// To achive a static shape the trait should be const implemented.
pub trait Shape<const NDIM: usize> {
    /// Length in the nth dimension.
    ///
    /// ## Example
    /// ```rust
    /// use slas::tensor::Shape;
    /// let s = slas::tensor::MatrixShape::<2, 3>;
    /// assert_eq!(s.axis_len(0), 3);
    /// ```
    ///
    /// For a matrix the height is specified before the width.
    /// axis_len(0) is should always be the width of a tensor.
    fn axis_len(&self, n: usize) -> usize;

    /// Total amount of elements in a tensor with shape.
    /// ## Example
    /// ```rust
    /// use slas::tensor::Shape;
    /// let s = slas::tensor::MatrixShape::<2, 3>;
    /// assert_eq!(s.volume(), 6);
    /// ```
    fn volume(&self) -> usize {
        (0..NDIM).map(|n| self.axis_len(n)).product()
    }
}

impl<const LEN: usize> Shape<LEN> for [usize; LEN] {
    fn axis_len(&self, n: usize) -> usize {
        self[n]
    }
}

/// Static matrix shape.
pub struct MatrixShape<const M: usize, const K: usize>;

impl<const M: usize, const K: usize> const Shape<2> for MatrixShape<M, K> {
    fn axis_len(&self, n: usize) -> usize {
        match n {
            0 => K,
            1 => M,
            _ => unsafe { unreachable_unchecked() },
        }
    }
    fn volume(&self) -> usize {
        M * K
    }
}

/// Statically allocated tensor.
/// See [`StaticVec::reshape`] for constructing a tensor.
/// The use of `&'static dyn Shape<NDIM>` does not mean slower performance,
/// as long as Shape is [const implemented](https://github.com/rust-lang/rust/issues/67792) for the type of the shape instance.
pub struct Tensor<T, U: StaticVec<T, LEN>, B: Backend<T>, const NDIM: usize, const LEN: usize> {
    pub data: WithStaticBackend<T, U, B, LEN>,
    pub shape: &'static dyn Shape<NDIM>,
}

impl<T: Float + std::fmt::Debug, B: Backend<T>, U: StaticVec<T, LEN>, const LEN: usize>
    std::fmt::Debug for Tensor<T, U, B, 2, LEN>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[\n")?;
        let m = self.shape.axis_len(1);
        let k = self.shape.axis_len(0);
        debug_assert_eq!(m * k, LEN);

        for n in 0..m {
            f.write_str("   ")?;
            f.debug_list()
                .entries((n * k..(n + 1) * k).map(|n| unsafe { self.data.data.get_unchecked(n) }))
                .finish()?;
            f.write_str(",\n")?;
        }
        f.write_str("]")?;
        Ok(())
    }
}

fn debug_shape<const NDIM: usize>(s: &dyn Shape<NDIM>) -> String {
    (0..NDIM)
        .map(|n| s.axis_len(n).to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn tensor_index<T: Shape<NDIM>, const NDIM: usize>(s: &dyn Shape<NDIM>, o: &T) -> usize {
    let mut sum = 0;
    let mut product = 1;
    for n in 0..NDIM {
        let i = o.axis_len(n);
        let j = s.axis_len(n);
        assert!(
            i < j,
            "Index {} out of bounds {}",
            debug_shape(o),
            debug_shape(s)
        );
        sum += i * product;
        product *= j;
    }
    sum
}

impl<
        T,
        U: StaticVec<T, LEN>,
        B: Backend<T>,
        S: Shape<NDIM>,
        const NDIM: usize,
        const LEN: usize,
    > std::ops::Index<S> for Tensor<T, U, B, NDIM, LEN>
{
    type Output = T;
    fn index(&self, i: S) -> &T {
        unsafe { self.data.get_unchecked(tensor_index(self.shape, &i)) }
    }
}
impl<
        T,
        U: StaticVec<T, LEN>,
        B: Backend<T>,
        S: Shape<NDIM>,
        const NDIM: usize,
        const LEN: usize,
    > std::ops::IndexMut<S> for Tensor<T, U, B, NDIM, LEN>
where
    T: Copy,
{
    fn index_mut(&mut self, i: S) -> &mut T {
        unsafe { self.data.get_unchecked_mut(tensor_index(self.shape, &i)) }
    }
}

impl<
        T: Float + Sized,
        U: StaticVec<T, LEN>,
        B: Backend<T> + operations::MatrixMul<T>,
        const LEN: usize,
    > Tensor<T, U, B, 2, LEN>
{
    pub fn matrix_mul_buffer<
        U2: StaticVec<T, LEN2>,
        U3: StaticVec<T, OLEN>,
        const LEN2: usize,
        const OLEN: usize,
    >(
        &self,
        other: &Tensor<T, U2, B, 2, LEN2>,
        buffer: &mut U3,
    ) {
        let m = self.shape.axis_len(1);
        let k = self.shape.axis_len(0);
        let n = other.shape.axis_len(0);

        debug_assert_eq!(self.shape.volume(), LEN);
        debug_assert_eq!(other.shape.volume(), LEN2);
        debug_assert_eq!(m * n, OLEN);

        <B as Backend<T>>::matrix_mul(
            &self.data.backend,
            &self.data.data,
            &other.data.data,
            buffer,
            m,
            n,
            k,
        );
    }

    pub fn matrix_mul<U2: StaticVec<T, LEN2>, const LEN2: usize, const OLEN: usize>(
        &self,
        other: &Tensor<T, U2, B, 2, LEN2>,
    ) -> [T; OLEN] {
        let mut buffer = [T::zero(); OLEN];
        self.matrix_mul_buffer(other, &mut buffer);
        buffer
    }
}

#[macro_export]
macro_rules! m {
    ($m: expr, $k: expr) => {
        MatrixShape::<$m, $k>
    };
}
