use crate::prelude::*;
use std::simd::Simd;

pub mod operations {
    use super::*;

    /// Dot product on a given backend.
    pub trait DotProduct {
        fn sdot<const LEN: usize>(
            &self,
            a: &impl StaticVec<f32, LEN>,
            b: &impl StaticVec<f32, LEN>,
        ) -> f32;
        fn ddot<const LEN: usize>(
            &self,
            a: &impl StaticVec<f64, LEN>,
            b: &impl StaticVec<f64, LEN>,
        ) -> f64;
    }

    /// Dot product for complex numbers on a given backend.
    pub trait ComplexDotProduct {
        fn cdotu_sub<const LEN: usize>(
            &self,
            a: &impl StaticVec<Complex<f32>, LEN>,
            b: &impl StaticVec<Complex<f32>, LEN>,
        ) -> Complex<f32>;
        fn zdotu_sub<const LEN: usize>(
            &self,
            a: &impl StaticVec<Complex<f64>, LEN>,
            b: &impl StaticVec<Complex<f64>, LEN>,
        ) -> Complex<f64>;
    }
}

macro_rules! impl_default_ops {
    ($float: ty,  $comp_blas_fn: ident, $slas_fn: ident) => {
        impl<'a, const LEN: usize> StaticVecUnion<'a, Complex<$float>, LEN> {
            /// Dot product for two complex vectors using blas.
            /// There is no rust backend for complex dot products at the moment.
            pub fn dot(&self, other: &Self) -> Complex<$float> {
                Blas.$comp_blas_fn(self, other)
            }
        }

        impl<'a, const LEN: usize> StaticVecUnion<'a, $float, LEN> {
            /// Vector dot product.
            ///
            /// This can be called on any [`StaticVec`], by calling [`StaticVec::moo_ref`] on it first.
            ///
            /// ## Example
            /// ```rust
            /// use slas::prelude::*;
            /// assert_eq!(moo![f32: 0..4].dot(&[1.2; 4].moo()), 7.2)
            /// ```
            pub fn dot(&self, other: &Self) -> $float {
                use operations::*;
                if LEN > 750 {
                    Blas.$slas_fn(self, other)
                } else {
                    Rust.$slas_fn(self, other)
                }
            }
        }
    };
}

use crate::StaticVecUnion;
impl_default_ops!(f32, cdotu_sub, sdot);
impl_default_ops!(f64, zdotu_sub, ddot);

mod blas;
pub use blas::*;

mod rust;
pub use rust::*;
