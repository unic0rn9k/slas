//! A slas backend defines how a set of supported algebraic operations can be performed on a specific software and/or hardware configuration.
//!
//! ## [`operations`]
//!
//! The first argument of an operations, should be a reference to an instance of the backend to run the operation on.
//! The following arguments should just be the arguments of the operation.
//!
//! The possible operations that can be implemented for a backend and its associated functions are:
//!
//! ### [`operations::DotProduct`]
//! Implemented for complex and real floats on [`slas_backend::Blas`].
//!
//! Implemented for real floats on [`slas_backend::Rust`].
//!
//! #### dot
//! Should take two vectors of equal length, and return their dot product.
//!
//! ### [`operations::Normalize`]
//! Implemented for real floats on [`slas_backend::Rust`].
//!
//! #### norm
//! Should return the euclidean length of a vector.
//!
//! #### normalize
//! Should normalize self (devide each element by the norm of the vector)
//!
//! ### [`operations::MatrixMul`]
//! Implemented for f32 and f64 -floats on [`slas_backend::Blas`].
//!
//! #### matrix_mul
//! Matrix-Matrix multiplication
//!
//! #### vector_mul
//! Matrix-Vector multiplication
//!
//! ### [`operations::Transpose`]
//!
//!
//! ## How to specify backend
//!
//! If you're trying to use slas on a system where blas isn't available,
//! you can use the [`slas_backend::Rust`] statically.
//!
//! ```rust
//! use slas::prelude::*;
//!
//! assert_eq!(
//!     moo![on slas_backend::Rust:f32: 0..4]
//!         .dot(&[1., 2., 3., 4.].moo_ref().static_backend()),
//!     20.
//! );
//! ```
//!
//! # Custom backend example
//! ```rust
//! use slas::prelude::*;
//! use slas::backends::operations;
//!
//! #[derive(Default)]
//! pub struct CustomBackend;
//!
//! impl<T: Float + std::iter::Sum> operations::DotProduct<T> for CustomBackend {
//!     fn dot<const LEN: usize>(
//!         &self,
//!         a: &impl StaticVec<T, LEN>,
//!         b: &impl StaticVec<T, LEN>,
//!     ) -> T {
//!         a.moo_ref().iter().zip(b.moo_ref().iter()).map(|(&a, &b)| a * b).sum()
//!     }
//! }
//!
//! impl<T> Backend<T> for CustomBackend{}
//! ```

use std::marker::PhantomData;

use crate::prelude::*;
use paste::paste;

macro_rules! impl_operations {
	($_t:ident $($name: ident $($op: ident ($($generics: tt)*) ($($generics_use: tt)*) ($($arg: ident : $arg_ty: ty),*)
        where ($($where_ty:ty : $implements: path),*)  -> $t: ty),*);*;) => {

        pub trait Backend<$_t>: Default{
            $($(
                fn $op<$($generics)*>(&self, $($arg : $arg_ty),*) -> paste!( <Self as operations::$name<$_t>>::[<$op:camel Output>] )
                where
                    Self: operations::$name<$_t>, $($where_ty : $implements),*
                {
                    <Self as operations::$name<$_t>>::$op::<$($generics_use)*>(self, $($arg),*)
                }
            )*)*
        }
        pub mod operations{
            use super::*;

            $(pub trait $name<$_t>{
                $(
                    paste!( type [<$op:camel Output>] = $t; );
                    fn $op<$($generics)*>(&self, $($arg : $arg_ty),*) -> paste!(Self::[<$op:camel Output>]) where $($where_ty : $implements),*;
                )*
            })*
        }
	};
}

impl_operations!(T
    DotProduct
        dot(const LEN: usize)()(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>
        ) where () -> T;

    Normalize
        norm(const LEN: usize)()(a: &impl StaticVec<T, LEN>) where () -> <Self as operations::Normalize<T>>::NormOutput,
        normalize(const LEN: usize)()(a: &mut impl StaticVec<T, LEN>) where (T: From<<Self as operations::Normalize<T>>::NormOutput>) -> ();

    MatrixMul
        matrix_mul(A: StaticVec<T, ALEN>, B: StaticVec<T, BLEN>, C: StaticVec<T, CLEN>, const ALEN: usize, const BLEN: usize, const CLEN: usize)
        (A, B, C, ALEN, BLEN, CLEN)
        (a: &A, b: &B, buffer: &mut C, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, a_trans: bool, b_trans: bool)
        where (
            A: Sized,
            B: Sized,
            C: Sized,
            T: Copy
        ) -> (),
        matrix_vector_mul(A: StaticVec<T, ALEN>, B: StaticVec<T, BLEN>, C: StaticVec<T, CLEN>, const ALEN: usize, const BLEN: usize, const CLEN: usize)
        (A, B, C, ALEN, BLEN, CLEN)
        (a: &A, b: &B, buffer: &mut C, m: usize, n: usize, lda: usize, a_trans: bool)
        where (
            A: Sized,
            B: Sized,
            C: Sized,
            T: Copy
        ) -> ();

    Transpose
        transpose_inplace(const LEN: usize)()(a: &mut impl StaticVec<T, LEN>, columns: usize) where () -> (),
        transpose(const LEN: usize)()(a: &impl StaticVec<T, LEN>, buffer: &mut impl StaticVec<T, LEN>, columns: usize) where () -> ();

    Addition
        add(const LEN: usize)()(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>,
            c: &mut impl StaticVec<T, LEN>
        ) where () -> ();

    Subtraction
        sub(const LEN: usize)()(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>,
            c: &mut impl StaticVec<T, LEN>
        ) where () -> ();

     Multiplication
        mul(const LEN: usize)()(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>,
            c: &mut impl StaticVec<T, LEN>
        ) where () -> ();

     Divition
        div(const LEN: usize)()(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>,
            c: &mut impl StaticVec<T, LEN>
        ) where () -> ();
);

/// Perform opertaions on a [`StaticVec`] with a static backend.
#[derive(Clone, Copy)]
pub struct WithStaticBackend<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> {
    pub data: U,
    pub backend: B,
    pub _pd: PhantomData<T>,
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> std::ops::Deref
    for WithStaticBackend<T, U, B, LEN>
{
    type Target = U;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> std::ops::DerefMut
    for WithStaticBackend<T, U, B, LEN>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> WithStaticBackend<T, U, B, LEN> {
    pub const fn from_static_vec(v: U, b: B) -> Self {
        Self {
            data: v,
            backend: b,
            _pd: PhantomData,
        }
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> StaticVec<T, LEN>
    for WithStaticBackend<T, U, B, LEN>
{
    unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> WithStaticBackend<T, U, B, LEN> {
    pub fn matrix<const M: usize, const K: usize>(
        self,
    ) -> crate::tensor::Matrix<T, U, B, LEN, false, MatrixShape<M, K>>
    where
        Self: Sized,
    {
        self.data
            .reshape(MatrixShape::<M, K>, self.backend)
            .matrix()
    }

    pub fn reshape<S: crate::tensor::Shape<NDIM>, const NDIM: usize>(
        self,
        shape: S,
    ) -> crate::tensor::Tensor<T, U, B, NDIM, LEN, S>
    where
        Self: Sized,
    {
        self.data.reshape(shape, self.backend)
    }
}

macro_rules! impl_default_ops {
    ($t: ty) => {
        impl<'a, const LEN: usize> StaticVecUnion<'a, Complex<$t>, LEN> {
            /// Dot product for two complex vectors using blas.
            /// There is no rust backend for complex dot products at the moment.
            pub fn dot(&self, other: &Self) -> Complex<$t> {
                Blas.dot(self, other)
            }
        }

        impl<'a, const LEN: usize> StaticVecUnion<'a, $t, LEN> {
            /// Vector dot product.
            ///
            /// This can be called on any [`StaticVec`], by calling [`StaticVec::moo_ref`] on it first.
            ///
            /// ## Example
            /// ```rust
            /// use slas::prelude::*;
            ///
            /// // There is some inaccuracy due to float rounding.
            /// assert!(moo![f32: 0..4].dot([1.2; 4].moo_ref()) - 7.2 < 0.000003)
            /// ```
            pub fn dot(&self, other: &Self) -> $t {
                if LEN >= crate::config::BLAS_IN_DOT_IF_LEN_GE {
                    Blas.dot(self, other)
                } else {
                    Rust.dot(self, other)
                }
            }
        }
    };
}

use crate::tensor::MatrixShape;
use crate::StaticVecUnion;
impl_default_ops!(f32);
impl_default_ops!(f64);

impl<'a, T: Float + std::iter::Sum, const LEN: usize> StaticVecUnion<'a, T, LEN>
where
    Rust: Backend<T>,
    Rust: operations::Normalize<T>,
    T: From<<Rust as operations::Normalize<T>>::NormOutput>,
{
    /// Normalize vector. Uses rust by default, as Normalize is not implemented for blas yet.
    pub fn normalize(&mut self) {
        Rust.normalize(self);
    }

    /// Returns norm of vector. Uses rust by default, as Normalize is not implemented for blas yet.
    pub fn norm(&mut self) -> <Rust as operations::Normalize<T>>::NormOutput {
        Rust.norm(self)
    }
}

impl<
        T,
        U: StaticVec<T, LEN>,
        B: Backend<T> + operations::DotProduct<T, DotOutput = T>,
        const LEN: usize,
    > WithStaticBackend<T, U, B, LEN>
{
    pub fn dot<U2: StaticVec<T, LEN>>(&self, other: &WithStaticBackend<T, U2, B, LEN>) -> T {
        operations::DotProduct::<T>::dot(&self.backend, &self.data, &other.data)
    }
}
impl<
        T: From<NormOutput>,
        NormOutput,
        U: StaticVec<T, LEN>,
        B: Backend<T> + operations::Normalize<T, NormOutput = NormOutput>,
        const LEN: usize,
    > WithStaticBackend<T, U, B, LEN>
{
    pub fn norm(&self) -> NormOutput {
        operations::Normalize::<T>::norm(&self.backend, &self.data)
    }
    pub fn normalize(&mut self) {
        operations::Normalize::<T>::normalize(&self.backend, &mut self.data);
    }
}

mod blas;
pub use blas::Blas;

mod rust;
pub use rust::Rust;
