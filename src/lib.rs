//! [![Workflow Status](https://github.com/unic0rn9k/slas/actions/workflows/tests.yml/badge.svg)](https://github.com/unic0rn9k/slas/actions/workflows/tests.yml)
//! Static Linear Algebra System. A object oriented expansion of blas/blis, that allow for statically allocated cow (Copy On Write) vectors, matricies and tensors.
//!
//! ## Example
//! ```rust
//! use slas::prelude::*;
//! let a = moo![f32: 1, 2, 3.2];
//! let b = moo![f32: 3, 0.4, 5];
//! println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
//! ```

#![allow(incomplete_features)]
#![feature(adt_const_params, generic_const_exprs)]

//mod experimental {
//    mod matrix;
//    mod tensor;
//}
mod matrix_stable;
pub use matrix_stable::matrix;
pub mod prelude;

use num::*;
use std::{convert::TryInto, fmt::Write, hint::unreachable_unchecked, ops::*};
extern crate blas_src;
extern crate cblas_sys;

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

    pub fn norm(&mut self)
    where
        T: Float + std::iter::Sum,
    {
        let len = self.iter().map(|n| n.powi(2)).sum::<T>().sqrt();
        self.iter_mut().for_each(|n| *n = *n / len);
    }
}

macro_rules! impl_dot {
    ($float: ty, $blas_fn: ident) => {
        impl<'a, const LEN: usize> StaticCowVec<'a, $float, LEN> {
            pub fn dot<'b>(&self, other: &Self) -> $float {
                unsafe { cblas_sys::$blas_fn(LEN as i32, self.as_ptr(), 1, other.as_ptr(), 1) }
            }
        }
    };
}
macro_rules! impl_dot_complex {
    ($float: ty, $blas_fn: ident) => {
        impl<'a, const LEN: usize> StaticCowVec<'a, Complex<$float>, LEN> {
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
        assert!(s.len() == LEN);
        Self::Borrowed(s)
    }
}

impl<'a, T: NumCast + Copy + std::fmt::Debug, const LEN: usize> std::fmt::Debug
    for StaticCowVec<'a, T, LEN>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_borrowed() {
            f.write_char('&')?;
        }
        f.debug_list().entries(self.iter()).finish()
    }
}

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
