//! A slas backend defines how a set of supported algebraic operations can be performed on a specific software and/or hardware configuration.
//!
//! ## Operations
//!
//! The possible operations that can be implemented for a backend and its associated functions are
//!
//! ### DotProduct
//! Dot product for regular floats.
//!
//! **For 32-bit floats** :
//! ```rust
//! fn sdot(const LEN: usize)(
//!     &self,
//!     a: &impl StaticVec<f32, LEN>,
//!     b: &impl StaticVec<f32, LEN>
//! ) -> f32
//! ```
//!
//! **For 64-bit floats** :
//! ```rust
//! fn ddot(const LEN: usize)(
//!     &self,
//!     a: &impl StaticVec<f64, LEN>,
//!     b: &impl StaticVec<f64, LEN>
//! ) -> f64
//! ```
//!
//! ### ComplexDotProduct
//! Dot product for complex numbers.
//!
//! **For 32-bit complex numbers** :
//! ```rust
//! fn cdotu_sub(const LEN: usize)(
//!     &self,
//!     a: &impl StaticVec<Complex<f32>, LEN>,
//!     b: &impl StaticVec<Complex<f32>, LEN>
//! ) -> Complex<f32>,
//! ```
//!
//! **For 32-bit complex numbers** :
//! ```rust
//! fn zdotu_sub(const LEN: usize)(
//!     &self,
//!     a: &impl StaticVec<Complex<f64>, LEN>,
//!     b: &impl StaticVec<Complex<f64>, LEN>
//! ) -> Complex<f64>;
//! ```
//!
//! ## How to specify backend
//!
//! If you're trying to use slas on a system where blas isn't available,
//! you can use the `slas_backend::Rust` statically.
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
	($_t:ident : $($name: ident $($op: ident ($($generics: tt)*) ($($arg: ident : $arg_ty: ty),*) -> $t: ty),*);*;) => {
        pub trait Backend<$_t>: Default{
            $($(
                fn $op<$($generics)*>(&mut self, $($arg : $arg_ty),*) -> $t where Self: operations::$name<$_t>{
                    <Self as operations::$name<$_t>>::$op(self, $($arg),*)
                }
            )*)*
        }
        pub mod operations{
            use super::*;

            $(pub trait $name<$_t>{
                $(fn $op<$($generics)*>(&mut self, $($arg : $arg_ty),*) -> $t;)*
            })*
        }
	};
}

impl_operations!(T:
    DotProduct
        dot(const LEN: usize)(
            a: &impl StaticVec<T, LEN>,
            b: &impl StaticVec<T, LEN>
        ) -> T;
);

#[derive(Clone, Copy)]
pub struct WithStaticBackend<T, U: StaticVec<T, LEN>, B: Backend<T>, const LEN: usize> {
    pub data: U,
    pub backend: B,
    pub _pd: PhantomData<T>,
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
                    Blas.dot(self, other)
                } else {
                    Rust.dot(self, other)
                }
            }
        }
    };
}

macro_rules! impl_with_backend {
    ($($op: ident : $fn: ident : $t: ty),*) => {$(
        impl<U: StaticVec<$t, LEN>, B: Backend<$t> + operations::$op<$t>, const LEN: usize>
            WithStaticBackend<$t, U, B, LEN>
        {
            pub fn dot<U2: StaticVec<$t, LEN>>(
                &mut self,
                other: &WithStaticBackend<$t, U2, B, LEN>,
            ) -> $t {
                operations::$op::<$t>::$fn(&mut self.backend, &self.data, &other.data)
            }
        }
    )*};
}

use crate::StaticVecUnion;
impl_default_ops!(f32);
impl_default_ops!(f64);
impl_with_backend![
    DotProduct: dot: f32,
    DotProduct: dot: f64,
    DotProduct: dot: Complex<f64>,
    DotProduct: dot: Complex<f32>
];

mod blas;
pub use blas::Blas;

mod rust;
pub use rust::Rust;
