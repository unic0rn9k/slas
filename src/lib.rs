//! <div align="center">
//!
//! # SLAS
//! *Static Linear Algebra System*
//!
//! [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/slas/Tests?label=tests&style=flat-square)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)
//! [![Donate on paypal](https://img.shields.io/badge/paypal-donate-1?style=flat-square&logo=paypal&color=blue)](https://www.paypal.com/paypalme/unic0rn9k/5usd)
//!
//! </div>
//!
//! Provides statically allocated vector, matrix and tensor types, for interfacing with blas/blis, in a performant manor, using copy-on-write (aka cow) behavior.
//!
//! ## Example
//!
//! ```rust
//! use slas::prelude::*;
//! let a = moo![f32: 1, 2, 3.2];
//! let b = moo![f32: 3, 0.4, 5];
//! println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
//! ```
//! [More example code here.](https://github.com/unic0rn9k/slas/blob/master/tests/src/main.rs)
//!
//! ## What is a COW?
//! The copy-on-write functionality is inspired by [std::borrow::cow](https://doc.rust-lang.org/std/borrow/enum.Cow.html).
//! The idea is simply that allocations (and time) can be saved, by figuring out when to copy at runtime instead of at compiletime.
//! This can be memory ineficient at times (as an enum takes the size of its largest field + tag size), but I'm planing on making ways around this in the future.
//!
//! ### In code...
//! ```rust
//! let source: Vec<f32> = vec![1., 2., 3.];
//! let mut v = moo![_ source.as_slice()];
//!
//! // Here we mutate v,
//! // so the content of source will be copied into v before the mutation occours.
//! v[0] = 0.;
//!
//! assert_eq!(*v, [0., 2., 3.]);
//! assert_eq!(source, vec![1., 2., 3.]);
//! ```
//!
//! The borrow checker won't allow mutating `source` after `v` is created, because assignment to borrowed values is not allowed.
//! This can be a problem in some situations.
//!
//! ```rust
//! let mut source: Vec<f32> = vec![1., 2., 3.];
//! let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };
//!
//! // Here we can mutate source, because v was created from a raw pointer.
//! source[1] = 3.;
//! v[0] = 0.;
//! source[2] = 4.;
//!
//! assert_eq!(*v, [0., 3., 3.]);
//! assert_eq!(source, vec![1., 3., 4.]);
//! ```
//! In the example above, you can see `v` changed value the first time `source` was mutated, but not the second time.
//! This is because `v` was copied when it was mutated at the line after the first mutation of `source`.
//!
//! ## Matricies, tensors and other mathematical types
//! At the moment the way I want to implement these types, causes a compiler crash, when trying to create 2 objects with the same shape.
//! For now I'm going to try to create a temporary, and more stable, way of dealing with these variations of static multi dimensional arrays.
//!
//! As of now there is a Matrix type, but no tensor type on the master branch.
//! The stable matricies are very basic, as I hopefully will be able to replace them with a more generic tensor type soon...
//!
//! ```rust
//!  use slas::{prelude::*, matrix::Matrix};
//!
//!  let m: Matrix<f32, 2, 3> = [
//!   1., 2., 3.,
//!   4., 5., 6.
//!  ].into();
//!
//!  assert!(m[[1, 0]] == 2.);
//!
//!  let k: Matrix<f32, 3, 2> = moo![f32: 0..6].into();
//!
//!  println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
//! ```
//!
//! If you want a look at whats to come in the future,
//! you can go [here](https://github.com/unic0rn9k/slas/tree/experimental/src/experimental)
//! for some *very* experimental source code for the project.
//!
//! ## Why not just use ndarray (or alike)?
//! Slas can be faster than ndarray in some specifik use cases, like when having to do a lot of allocations, or when using referenced data in vector operations.
//! Besides slas should always be atleast as fast as ndarray, so it can't hurt.
//!
//! Statcal allocation and the way slas cow behavior works with the borrow checker,
//! also means that you might catch a lot of bugs at compiletime,
//! where ndarray most of the time will let you get away with pretty much anything.
//!
//! ## More info...
//! - Slas is still in very early days, and is subject to a lot of breaking changes.
//! - The `StaticCowVec` type implements `deref` and `deref_mut`, so any method implemented for `[T;LEN]` is also implemented for `StaticCowVec`.
//! - [Benchmarks, tests and related](https://github.com/unic0rn9k/slas/tree/master/tests)
//!
//! ## TODO: before publishing 🎉
//! - ~~Move ./experimental to other branch~~
//! - ~~Implement Debug for matrix~~
//! - ~~Fix matrix api (Column and row specification is weird)~~
//! - ~~Write documentation~~
//! - Benchmark against ndarray - and maybe others? numpy?
//!
//! ## TODO: after publish
//! - Feature support for conversion between [ndarray](lib.rs/ndarray) types
//! - Allow for use on stable channel - perhabs with a stable feature
//! - Implement stable tensors - perhabs for predefined dimensions with a macro
//! - Make StaticCowVec backed by a union -so that vectors that are always owned can also be supported (useful for memory critical systems, fx. embeded devices).
//!
//! ## TODO: Long term
//! - GPU support - maybe with cublas

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod matrix_stable;
pub use matrix_stable::matrix;
pub mod prelude;

use num::*;
use std::{convert::TryInto, hint::unreachable_unchecked, ops::*};
extern crate blas_src;
extern crate cblas_sys;

/// Statically allocated copy-on-write vector struct.
/// This is the backbone of the crate, and is also the type used inside of matricies and tensors.
#[derive(Clone, Copy)]
pub enum StaticCowVec<'a, T: NumCast + Copy, const LEN: usize> {
    Owned([T; LEN]),
    Borrowed(&'a [T]),
}

impl<'a, T: NumCast + Copy, const LEN: usize> StaticCowVec<'a, T, LEN> {
    pub fn zeros() -> Self {
        Self::Owned([T::from(0).unwrap(); LEN])
    }

    pub fn len(&self) -> usize {
        LEN
    }

    pub fn is_borrowed(&self) -> bool {
        match self {
            Self::Borrowed(_) => true,
            _ => false,
        }
    }

    pub fn is_owned(&self) -> bool {
        !self.is_borrowed()
    }

    /// Slow, quick and dirty norm function, which normalizes a vector.
    pub fn norm(&mut self)
    where
        T: Float + std::iter::Sum,
    {
        // TODO: Make me fast. Use blas.
        let len = self.iter().map(|n| n.powi(2)).sum::<T>().sqrt();
        self.iter_mut().for_each(|n| *n = *n / len);
    }

    pub unsafe fn from_ptr(ptr: *const T) -> Self {
        Self::Borrowed((ptr as *const [T; LEN]).as_ref().unwrap())
    }
}

macro_rules! impl_dot {
    ($float: ty, $blas_fn: ident) => {
        impl<'a, const LEN: usize> StaticCowVec<'a, $float, LEN> {
            /// Dot product for two vectors of same length using blas.
            pub fn dot<'b>(&self, other: &Self) -> $float {
                unsafe { cblas_sys::$blas_fn(LEN as i32, self.as_ptr(), 1, other.as_ptr(), 1) }
            }
        }
    };
}
macro_rules! impl_dot_complex {
    ($float: ty, $blas_fn: ident) => {
        impl<'a, const LEN: usize> StaticCowVec<'a, Complex<$float>, LEN> {
            /// Dot product for two complex vectors of same length using blas.
            pub fn dot<'b>(&self, other: &Self) -> Complex<$float> {
                let mut tmp: [$float; 2] = [0.; 2];
                unsafe {
                    cblas_sys::$blas_fn(
                        LEN as i32,
                        self.as_ptr() as *const [$float; 2],
                        1,
                        other.as_ptr() as *const [$float; 2],
                        1,
                        tmp.as_mut_ptr() as *mut [$float; 2],
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

impl_dot!(f32, cblas_sdot);
impl_dot!(f64, cblas_ddot);
impl_dot_complex!(f32, cblas_cdotu_sub);
impl_dot_complex!(f64, cblas_zdotu_sub);

impl<'a, T: NumCast + Copy, const LEN: usize> Deref for StaticCowVec<'a, T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(b) => unsafe { (*b).try_into().unwrap_unchecked() }, // I'm afraid to cast raw pointer here, since I think it might cause incorrect lifetimes.
        }
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> DerefMut for StaticCowVec<'a, T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(b) => {
                *self = Self::Owned(unsafe { *(b.as_ptr() as *mut [T; LEN]) });
                match self {
                    Self::Owned(o) => o,
                    _ => unsafe { unreachable_unchecked() },
                }
            }
        }
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> From<[T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: [T; LEN]) -> Self {
        Self::Owned(s)
    }
}
impl<'a, T: NumCast + Copy, const LEN: usize> From<&'a [T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T; LEN]) -> Self {
        Self::Borrowed(&s[..])
    }
}
impl<'a, T: NumCast + Copy, const LEN: usize> From<&'a [T]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T]) -> Self {
        assert_eq!(s.len(), LEN); // TODO: Unchecked version of this...
        Self::Borrowed(s)
    }
}

impl<'a, T: NumCast + Copy + std::fmt::Debug, const LEN: usize> std::fmt::Debug
    for StaticCowVec<'a, T, LEN>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        if self.is_borrowed() {
            f.write_char('&')?;
        }
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Macro for creating [`StaticCowVec`]'s
///
/// ## Example
/// ```rust
/// moo![f32: 1, 2, 3.5];
/// moo![f32: 1..4];
/// moo![f32: 1..=3];
/// moo![0f32; 4];
/// moo![_ source.as_slice()];
/// ```
#[macro_export]
macro_rules! moo {
    (_ $($v: tt)*) => {{
        StaticCowVec::from($($v)*)
    }};
    ($t: ty: $a: literal .. $b: literal) => {{
        let mut tmp = StaticCowVec::from([0 as $t; $b - $a]);
        tmp.iter_mut().zip($a..$b).for_each(|(o, i)| *o = i as $t);
        tmp
    }};
    ($t: ty: $a: literal ..= $b: literal) => {{
        let mut tmp = StaticCowVec::from([0 as $t; $b - $a+1]);
        tmp.iter_mut().zip($a..=$b).for_each(|(o, i)| *o = i as $t);
        tmp
    }};
    ($t: ty: $($v: expr),* $(,)?) => {{
        StaticCowVec::from([$({$v} as $t),*])
    }};
    ($($v: tt)*) => {{
        StaticCowVec::from([$($v)*])
    }};
}
