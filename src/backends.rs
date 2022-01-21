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
//! Implemented for real floats on [`slas_backend::Blas`].
//!
//! #### matrix_mul
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

use std::marker::PhantomData;

use crate::prelude::*;

macro_rules! impl_operations {
	($_t:ident $($name: ident $($op: ident ($($generics: tt)*) ($($generics_use: tt)*) ($($arg: ident : $arg_ty: ty),*) where ($($where_ty:ty : $implements: tt),*)  -> $t: ty),*);*;) => {
        pub trait Backend<$_t>: Default{
            $($(
                fn $op<$($generics)*>(&self, $($arg : $arg_ty),*) -> $t where Self: operations::$name<$_t>, $($where_ty : $implements),*{
                    <Self as operations::$name<$_t>>::$op::<$($generics_use)*>(self, $($arg),*)
                }
            )*)*
        }
        pub mod operations{
            use super::*;

            $(pub trait $name<$_t>{
                $(fn $op<$($generics)*>(&self, $($arg : $arg_ty),*) -> $t where $($where_ty : $implements),*;)*
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
        norm(const LEN: usize)()(a: &impl StaticVec<T, LEN>) where () -> T,
        normalize(const LEN: usize)()(a: &mut impl StaticVec<T, LEN>) where () -> ();

    MatrixMul
        matrix_mul(A: StaticVec<T, ALEN>, B: StaticVec<T, BLEN>, const ALEN: usize, const BLEN: usize, const OLEN: usize)
        (A, B, ALEN, BLEN, OLEN)
        (a: &A, b: &B, m: usize, n: usize, k: usize)
        where (
            A: Sized,
            B: Sized,
            T: Copy
        )
        ->  [T; OLEN];
);

/// Perform opertaions on a [`StaticVec`] with a static backend.
#[derive(Clone, Copy)]
pub struct WithStaticBackend<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> {
    pub data: U,
    pub backend: B,
    pub _pd: PhantomData<T>,
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
    pub fn matrix<const M: usize, const K: usize>(self) -> crate::tensor::Tensor<T, U, B, 2, LEN>
    where
        Self: Sized,
    {
        self.data.reshape(&MatrixShape::<M, K>, self.backend)
    }

    pub fn reshape<S: crate::tensor::Shape<NDIM>, const NDIM: usize>(
        self,
        shape: &'static S,
    ) -> crate::tensor::Tensor<T, U, B, NDIM, LEN>
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
                if LEN > 750 {
                    // FIXME: This should not always be 750.
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

impl<T, U: StaticVec<T, LEN>, B: Backend<T> + operations::DotProduct<T>, const LEN: usize>
    WithStaticBackend<T, U, B, LEN>
{
    pub fn dot<U2: StaticVec<T, LEN>>(&self, other: &WithStaticBackend<T, U2, B, LEN>) -> T {
        operations::DotProduct::<T>::dot(&self.backend, &self.data, &other.data)
    }
}
impl<T, U: StaticVec<T, LEN>, B: Backend<T> + operations::Normalize<T>, const LEN: usize>
    WithStaticBackend<T, U, B, LEN>
{
    pub fn norm(&self) -> T {
        operations::Normalize::<T>::norm(&self.backend, &self.data)
    }
    pub fn normalize(&mut self) {
        operations::Normalize::<T>::normalize(&mut self.backend, &mut self.data)
    }
}

mod blas;
pub use blas::Blas;

mod rust;
pub use rust::Rust;
