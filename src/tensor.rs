use std::{marker::PhantomData, mem::transmute, ops::Deref, ops::DerefMut};

use crate::{backends::*, prelude::*};
use paste::paste;

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

    /// Return shape as a slice
    fn slice(&self) -> &[usize; NDIM];
}

impl<const LEN: usize> Shape<LEN> for [usize; LEN] {
    fn axis_len(&self, n: usize) -> usize {
        self[n]
    }

    fn slice(&self) -> &[usize; LEN] {
        self
    }
}

impl<const LEN: usize> Shape<LEN> for [usize] {
    fn axis_len(&self, n: usize) -> usize {
        self[n]
    }

    fn slice(&self) -> &[usize; LEN] {
        assert_eq!(self.len(), LEN);
        unsafe { std::mem::transmute(self.as_ptr()) }
    }
}

/// Static matrix shape.
pub struct MatrixShape<const M: usize, const K: usize>;

impl<const M: usize, const K: usize> const Shape<2> for MatrixShape<M, K> {
    fn axis_len(&self, n: usize) -> usize {
        match n {
            0 => K,
            1 => M,
            _ => panic!("Cannot get len of axis higher than 1, as a matrix only has 2 axies (rows and columns)"),
        }
    }
    fn volume(&self) -> usize {
        M * K
    }
    fn slice(&self) -> &[usize; 2] {
        &[K, M]
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
    std::fmt::Debug for Matrix<T, U, B, LEN>
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
        unsafe { self.data.data.get_unchecked(tensor_index(self.shape, &i)) }
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
        unsafe {
            self.data
                .data
                .get_unchecked_mut(tensor_index(self.shape, &i))
        }
    }
}

macro_rules! impl_index_slice {
	($($mut: tt)?) => {
		impl<'a, T, U: StaticVec<T, LEN> + 'a, B: Backend<T>, const NDIM: usize, const LEN: usize>
            Tensor<T, U, B, NDIM, LEN>
        where
            [(); NDIM - 1]: Sized,
            &'a $($mut)? U: StaticVec<T, LEN>,
        {
            paste!{pub fn [<index_slice $(_$mut)?>] (&'a $($mut)? self, i: usize) -> Tensor<T, &'a $($mut)? [T; LEN], B, { NDIM - 1 }, LEN> {
                assert!(NDIM > 1);
                assert!(i < self.shape.axis_len(0));

                unsafe {
                    transmute::<*const T, &'a $($mut)? [T; LEN]>(
                        self.data
                            .[< as $(_$mut)? _ptr>]()
                            .add(i * (self.shape.volume() / self.shape.axis_len(NDIM - 1))),
                    )
                    .[<reshape_unchecked_ref $(_$mut)? >](
                        transmute::<*const usize, &[usize; NDIM - 1]>(
                            self.shape.slice()[0..NDIM - 1].as_ptr(),
                        ),
                        B::default(),
                    )
                }
            }}
        }
	};
}

impl_index_slice!();
impl_index_slice!(mut);

pub trait IsTrans {
    fn is_transposed(&self) -> bool;

    fn const_is_trans() -> bool
    where
        Self: ~const IsTrans;
}

pub trait Matrixish<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;

    fn moo_ref(&self) -> StaticVecRef<T, LEN>
    where
        T: Copy;

    fn mut_moo_ref(&mut self) -> MutStaticVecRef<T, LEN>
    where
        T: Copy;

    fn transpose(self) -> TransposedMatrix<T, U, B, Self, LEN, { !Self::const_is_trans() }>
    where
        Self: ~const IsTrans + Sized,
    {
        TransposedMatrix(self, PhantomData)
    }

    fn backend(&self) -> &B;

    fn matrix_mul_buffer<
        U2: StaticVec<T, LEN2>,
        U3: StaticVec<T, OLEN>,
        const LEN2: usize,
        const OLEN: usize,
    >(
        &self,
        other: &(impl Matrixish<T, U2, B, LEN2> + IsTrans),
        buffer: &mut U3,
    ) where
        B: operations::MatrixMul<T>,
        T: Copy + Float,
        Self: IsTrans,
    {
        let m = self.rows();
        let k = self.columns();
        let n = other.columns();

        debug_assert_eq!(self.rows() * self.columns(), LEN);
        debug_assert_eq!(other.rows() * other.columns(), LEN2);
        debug_assert_eq!(m * n, OLEN);

        <B as Backend<T>>::matrix_mul(
            self.backend(),
            self.moo_ref(),
            other.moo_ref(),
            buffer,
            m,
            n,
            k,
            self.is_transposed(),
            other.is_transposed(),
        );
    }

    fn matrix_mul<U2: StaticVec<T, LEN2>, const LEN2: usize, const OLEN: usize>(
        &self,
        other: &(impl Matrixish<T, U2, B, LEN2> + IsTrans),
    ) -> [T; OLEN]
    where
        B: operations::MatrixMul<T>,
        T: Copy + Float,
        Self: IsTrans,
    {
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

pub type Matrix<T, U, B, const LEN: usize> = Tensor<T, U, B, 2, LEN>;

pub struct TransposedMatrix<
    T,
    U: StaticVec<T, LEN>,
    B: Backend<T>,
    M: Matrixish<T, U, B, LEN>,
    const LEN: usize,
    const IS_TRANS: bool = false,
>(M, PhantomData<(T, U, B)>);

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool> Deref
    for TransposedMatrix<T, U, B, Matrix<T, U, B, LEN>, LEN, IS_TRANS>
{
    type Target = Matrix<T, U, B, LEN>;
    fn deref(&self) -> &Self::Target {
        if IS_TRANS {
            todo!()
        } else {
            &self.0
        }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool> DerefMut
    for TransposedMatrix<T, U, B, Matrix<T, U, B, LEN>, LEN, IS_TRANS>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        if IS_TRANS {
            todo!()
        } else {
            &mut self.0
        }
    }
}

impl<
        T,
        U: StaticVec<T, LEN>,
        B: Backend<T>,
        M: Matrixish<T, U, B, LEN>,
        const LEN: usize,
        const IS_TRANS: bool,
    > const IsTrans for TransposedMatrix<T, U, B, M, LEN, IS_TRANS>
{
    fn is_transposed(&self) -> bool {
        IS_TRANS
    }
    fn const_is_trans() -> bool {
        IS_TRANS
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> const IsTrans
    for Matrix<T, U, B, LEN>
{
    fn is_transposed(&self) -> bool {
        false
    }
    fn const_is_trans() -> bool {
        false
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> Matrixish<T, U, B, LEN>
    for Matrix<T, U, B, LEN>
{
    fn moo_ref(&self) -> StaticVecRef<T, LEN>
    where
        T: Copy,
    {
        self.data.data.moo_ref()
    }

    fn mut_moo_ref(&mut self) -> MutStaticVecRef<T, LEN>
    where
        T: Copy,
    {
        self.data.data.mut_moo_ref()
    }

    fn backend(&self) -> &B {
        &self.data.backend
    }

    fn rows(&self) -> usize {
        self.shape.axis_len(1)
    }
    fn columns(&self) -> usize {
        self.shape.axis_len(0)
    }
}

impl<
        T,
        U: StaticVec<T, LEN>,
        B: Backend<T>,
        M: Matrixish<T, U, B, LEN>,
        const LEN: usize,
        const IS_TRANS: bool,
    > Matrixish<T, U, B, LEN> for TransposedMatrix<T, U, B, M, LEN, IS_TRANS>
{
    fn rows(&self) -> usize {
        if IS_TRANS {
            self.0.columns()
        } else {
            self.0.rows()
        }
    }
    fn columns(&self) -> usize {
        if !IS_TRANS {
            self.0.columns()
        } else {
            self.0.rows()
        }
    }

    fn moo_ref(&self) -> StaticVecRef<T, LEN>
    where
        T: Copy,
    {
        self.0.moo_ref()
    }
    fn mut_moo_ref(&mut self) -> MutStaticVecRef<T, LEN>
    where
        T: Copy,
    {
        self.0.mut_moo_ref()
    }

    fn backend(&self) -> &B {
        self.0.backend()
    }
}
