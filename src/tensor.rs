use crate::{backends::*, prelude::*};
use paste::paste;
use std::mem::transmute;

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
    #[inline(always)]
    fn axis_len(&self, n: usize) -> usize {
        self[n]
    }

    #[inline(always)]
    fn slice(&self) -> &[usize; LEN] {
        assert_eq!(self.len(), LEN);
        unsafe { &*(self.as_ptr() as *const [usize; LEN]) }
    }
}

/// Static matrix shape.
pub struct MatrixShape<const M: usize, const K: usize>;

impl<const M: usize, const K: usize> const Shape<2> for MatrixShape<M, K> {
    #[inline(always)]
    fn axis_len(&self, n: usize) -> usize {
        match n {
            0 => K,
            1 => M,
            _ => panic!("Cannot get len of axis higher than 1, as a matrix only has 2 axies (rows and columns)"),
        }
    }
    #[inline(always)]
    fn volume(&self) -> usize {
        M * K
    }
    #[inline(always)]
    fn slice(&self) -> &[usize; 2] {
        &[K, M]
    }
}

/// Statically allocated tensor.
/// See [`StaticVec::reshape`] for constructing a tensor.
/// The use of `&'static dyn Shape<NDIM>` does not mean slower performance,
/// as long as Shape is [const implemented](https://github.com/rust-lang/rust/issues/67792) for the type of the shape instance.
#[derive(Clone, Copy)]
pub struct Tensor<T, U: StaticVec<T, LEN>, B: Backend<T>, const NDIM: usize, const LEN: usize> {
    pub data: WithStaticBackend<T, U, B, LEN>,
    pub shape: &'static dyn Shape<NDIM>,
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> Tensor<T, U, B, 2, LEN> {
    pub const fn matrix(self) -> Matrix<T, U, B, LEN> {
        Matrix(self)
    }
    pub const fn backend(&self) -> &B {
        &self.data.backend
    }
    pub const fn vec_ref(&self) -> &U {
        &self.data.data
    }
    pub const fn mut_vec_ref(&mut self) -> &mut U {
        &mut self.data.data
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> const std::ops::Index<()>
    for Tensor<T, U, B, 2, LEN>
{
    type Output = Matrix<T, U, B, LEN>;
    fn index<'a>(&'a self, _: ()) -> &'a Self::Output {
        unsafe { transmute(self) }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> const std::ops::IndexMut<()>
    for Tensor<T, U, B, 2, LEN>
{
    fn index_mut<'a>(&'a mut self, _: ()) -> &'a mut Self::Output {
        unsafe { transmute(self) }
    }
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

#[inline(always)]
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

    #[inline(always)]
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

impl<
        T: Float + Sized,
        U: StaticVec<T, LEN>,
        B: Backend<T> + operations::MatrixMul<T>,
        const LEN: usize,
    > Matrix<T, U, B, LEN>
{
    pub fn matrix_mul_buffer<
        U2: StaticVec<T, LEN2>,
        U3: StaticVec<T, OLEN>,
        const LEN2: usize,
        const OLEN: usize,
    >(
        &self,
        other: &Matrix<T, U2, B, LEN2>,
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
            false,
            false,
        );
    }

    pub fn matrix_mul<U2: StaticVec<T, LEN2>, const LEN2: usize, const OLEN: usize>(
        &self,
        other: &Matrix<T, U2, B, LEN2>,
    ) -> [T; OLEN] {
        let mut buffer = [num::num!(0); OLEN];
        <Self>::matrix_mul_buffer(self, other, &mut buffer);
        buffer
    }
}

#[macro_export]
macro_rules! m {
    ($m: expr, $k: expr) => {
        MatrixShape::<$m, $k>
    };
}

/// Just a type alias for a 2D tensor.
/// pub type Matrix<T, U, B, const LEN: usize> = Tensor<T, U, B, 2, LEN>;

#[derive(Clone, Copy)]
pub struct Matrix<
    T,
    U: StaticVec<T, LEN>,
    B: Backend<T>,
    const LEN: usize,
    const IS_TRANS: bool = false,
>(Tensor<T, U, B, 2, LEN>);

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool>
    Matrix<T, U, B, LEN, IS_TRANS>
{
    #[inline(always)]
    pub fn rows(&self) -> usize {
        if IS_TRANS {
            self.0.shape.axis_len(0)
        } else {
            self.0.shape.axis_len(1)
        }
    }

    #[inline(always)]
    pub fn columns(&self) -> usize {
        if IS_TRANS {
            self.0.shape.axis_len(1)
        } else {
            self.0.shape.axis_len(0)
        }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool> const
    std::ops::Deref for Matrix<T, U, B, LEN, IS_TRANS>
{
    type Target = Tensor<T, U, B, 2, LEN>;
    fn deref(&self) -> &Self::Target {
        if IS_TRANS {
            panic!("Cannot deref lazily transposed matrix immutably. Try using &self.deref_mut() instead")
        } else {
            &self.0
        }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool> const
    std::ops::DerefMut for Matrix<T, U, B, LEN, IS_TRANS>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        if IS_TRANS {
            todo!()
        } else {
            &mut self.0
        }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize, const IS_TRANS: bool> const
    From<Tensor<T, U, B, 2, LEN>> for Matrix<T, U, B, LEN, IS_TRANS>
{
    fn from(t: Tensor<T, U, B, 2, LEN>) -> Self {
        Matrix(t)
    }
}
