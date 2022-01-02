//! <div align="center">
//!
//! # SLAS
//! *Static Linear Algebra System*
//!
//! [![Crates.io](https://img.shields.io/crates/v/slas?logo=rust&style=flat-square)](https://crates.io/crates/slas)
//! [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/slas/Tests?label=tests&logo=github&style=flat-square)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)
//! [![Docs](https://img.shields.io/docsrs/slas/latest?logo=rust&style=flat-square)](https://docs.rs/slas/latest/slas/)
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
//! This can be memory inefficient at times (as an enum takes the size of its largest field + tag size), but I'm planing on making ways around this in the future.
//!
//! **If you're using the git version of slas, you can now use `VecRef`'s instead of `StaticCowVecs`, when you don't want the cow behavior.**
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
//! use slas::{prelude::*, matrix::Matrix};
//!
//! let m: Matrix<f32, 2, 3> = [
//!  1., 2., 3.,
//!  4., 5., 6.
//! ].into();
//!
//! assert!(m[[1, 0]] == 2.);
//!
//! let k: Matrix<f32, 3, 2> = moo![f32: 0..6].into();
//!
//! println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
//! ```
//!
//! If you want a look at whats to come in the future,
//! you can go [here](https://github.com/unic0rn9k/slas/tree/experimental/src/experimental)
//! for some *very* experimental source code for the project.
//!
//! ## Why not just use ndarray (or alike)?
//! Slas can be faster than ndarray in some specific use cases, like when having to do a lot of allocations, or when using referenced data in vector operations.
//! Besides slas should always be atleast as fast as ndarray, so it can't hurt.
//!
//! Statical allocation and the way slas cow behavior works with the borrow checker,
//! also means that you might catch a lot of bugs at compiletime,
//! where ndarray most of the time will let you get away with pretty much anything.
//!
//! ## Installation
//! Slas depends on blas, and currently only supports using blis.
//! In the future you will have to choose your own blas provider, and instructions for doing so will be added here.
//!
//! On the crates.io version of slas (v0.1.0 and 0.1.1) blis is compiled automatically.
//!
//! For now, if you want to use the git version of slas, you need to install blis on your system.
//! - On Arch linux `blis-cblas` v0.7.0 from the aur has been tested and works fine.
//! - On Debian you can simply run `apt install libblis-dev` to install.
//!
//! ## General info...
//! - Slas is still in very early days, and is subject to a lot of breaking changes.
//! - The `StaticCowVec` type implements `deref` and `deref_mut`, so any method implemented for `[T;LEN]` is also implemented for `StaticCowVec`.
//! - [Benchmarks, tests and related](https://github.com/unic0rn9k/slas/tree/master/tests)
//!
//! ## TODO
//! - Make less terrible benchmarks
//! - Feature support for conversion between [ndarray](lib.rs/ndarray) types
//! - Allow for use on stable channel - perhabs with a stable feature
//! - Implement stable tensors - perhabs for predefined dimensions with a macro
//! - ~~Make StaticCowVec backed by a union - so that vectors that are always owned can also be supported (useful for memory critical systems, fx. embeded devices).~~
//! - Modular backends - [like in coaster](https://github.com/spearow/juice/tree/master/coaster)
//!     - GPU support - maybe with cublas
//!     - pure rust support - usefull for irust and jupyter support.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod matrix_stable;
pub use matrix_stable::matrix;
pub mod prelude;

use num::*;
use std::{marker::PhantomData, ops::*};
extern crate blis_src;
extern crate cblas_sys;

/// StaticVectorUnion is always owned when it is not found in a StaticCowVec, therefore we have this type alias to make it less confisung when dealing with references to owned vectors.
pub type VecRef<'a, T, const LEN: usize> = &'a StaticVectorUnion<'a, T, LEN>;

/// A very general trait for anything that can be called a static vector (fx. `[T; LEN]`)
///
/// **Warning:** If self is not contiguous, it will cause undefined behaviour.
///
/// Why does StaticVector not allow mutable access to self?
///
/// Because there is no overhead casting to [`StaticVectorUnion::owned`] and calling methods on that instead.
pub trait StaticVector<T, const LEN: usize> {
    /// Return pointer to first element.
    unsafe fn as_ptr(&self) -> *const T;

    /// Return a reference to self with the type of [`StaticVectorUnion::owned`]
    fn moo_ref<'a>(&'a self) -> VecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { &*(self.as_ptr() as *const StaticVectorUnion<'a, T, LEN>) }
    }

    fn moo<'a>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy + NumCast,
    {
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }
}

macro_rules! dyn_cast_panic {
    ($a: expr, $b: expr) => {{
        if $a != $b {
            panic!(
                "Cannot cast a DynamicVector of len {}, to a StaticVector with len {}",
                $a, $b
            )
        }
    }};
}

/// Allow to pretend that dynamically sized vectors are statically sized.
/// See [`StaticVector`] for more information.
///
/// ## Example
/// ```rust
/// let a = vec![0f32, 1., 2., 3.];
/// let b = moo![f32: 0, -1, -2, 3];
///
/// assert!(cblas_sdot(&a.pretend_static(), &b) == 4.)
/// ```
pub trait DynamicVector<T> {
    fn len(&self) -> usize;
    unsafe fn as_ptr(&self) -> *const T;
    fn pretend_static<const LEN: usize>(&self) -> PretendStaticVector<'_, T, Self, LEN> {
        dyn_cast_panic!(self.len(), LEN);
        PretendStaticVector(self, PhantomData)
    }
    unsafe fn pretend_static_unchecked<const LEN: usize>(
        &self,
    ) -> PretendStaticVector<'_, T, Self, LEN> {
        PretendStaticVector(self, PhantomData)
    }

    /// Return a reference to self with the type of [`StaticVectorUnion::owned`]
    fn moo_ref<'a, const LEN: usize>(&'a self) -> VecRef<'a, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { &*(self.as_ptr() as *const StaticVectorUnion<'a, T, LEN>) }
    }

    fn moo<'a, const LEN: usize>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy + NumCast,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }
}

// TODO: Implement deref
// TODO: Move methods and implementations from StaticCowVec, StaticVector and StaticVectorUnion to be more correct.
/// Will always be owned, unless inside a [`StaticVector`]
#[derive(Clone, Copy)]
pub union StaticVectorUnion<'a, T: Copy, const LEN: usize> {
    pub owned: [T; LEN],
    pub borrowed: &'a [T; LEN],
}

impl<'a, T: Copy, const LEN: usize> StaticVector<T, LEN> for StaticVectorUnion<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        self.owned.as_ptr()
    }
}

impl<T, const LEN: usize> StaticVector<T, LEN> for [T; LEN] {
    unsafe fn as_ptr(&self) -> *const T {
        self as *const T
    }
}

/// See [`StaticVector`].
pub struct PretendStaticVector<'a, I, T: DynamicVector<I> + ?Sized, const LEN: usize>(
    &'a T,
    PhantomData<I>,
);

impl<'a, I, T: DynamicVector<I>, const LEN: usize> StaticVector<I, LEN>
    for PretendStaticVector<'a, I, T, LEN>
{
    unsafe fn as_ptr(&self) -> *const I {
        self.0.as_ptr()
    }
}

impl<T> DynamicVector<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn as_ptr(&self) -> *const T {
        self as *const [T] as *const T
    }
}

impl<T> DynamicVector<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn as_ptr(&self) -> *const T {
        &self[0] as *const T
    }
}

// /// Statically allocated copy-on-write vector struct.
// /// This is the backbone of the crate, and is also the type used inside of matricies and tensors.
// #[derive(Clone, Copy)]
// pub enum StaticCowVec<'a, T: NumCast + Copy, const LEN: usize> {
//     Owned([T; LEN]),
//     Borrowed(&'a [T; LEN]),
// }

#[derive(Clone, Copy)]
pub struct StaticCowVec<'a, T: Copy, const LEN: usize> {
    data: StaticVectorUnion<'a, T, LEN>,
    is_owned: bool,
}

impl<'a, T: NumCast + Copy, const LEN: usize> StaticCowVec<'a, T, LEN> {
    //pub fn zeros() -> Self {
    //    // TODO:Replace this function with fill function, and remove NumCast as trait dependecy for T.
    //    Self::Owned([T::from(0).unwrap(); LEN])
    //}

    pub fn len(&self) -> usize {
        LEN
    }

    pub fn is_borrowed(&self) -> bool {
        !self.is_owned()
    }

    pub fn is_owned(&self) -> bool {
        self.is_owned
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
        Self::from(
            (ptr as *const [T; LEN])
                .as_ref()
                .expect("Cannot create StaticCowVec::Borrowed from NULL"),
        )
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> StaticVector<T, LEN> for StaticCowVec<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        match self.is_owned {
            true => self.data.as_ptr(),
            false => self.data.borrowed as *const T,
        }
    }
}

macro_rules! impl_dot {
    ($float: ty, $blas_fn: ident, $comp_blas_fn: ident) => {
        /// Dot product for two vectors of same length using blas.
        pub fn $blas_fn<const LEN: usize>(
            a: &impl StaticVector<$float, LEN>,
            b: &impl StaticVector<$float, LEN>,
        ) -> $float {
            unsafe { cblas_sys::$blas_fn(LEN as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
        }

        impl<'a, const LEN: usize> StaticCowVec<'a, $float, LEN> {
            /// Dot product for two vectors of same length using blas.
            pub fn dot(&self, other: &Self) -> $float {
                $blas_fn(self, other)
            }
        }
        /// Dot product for two complex vectors of same length using blas.
        pub fn $comp_blas_fn<const LEN: usize>(
            a: &impl StaticVector<Complex<$float>, LEN>,
            b: &impl StaticVector<Complex<$float>, LEN>,
        ) -> Complex<$float> {
            let mut tmp: [$float; 2] = [0.; 2];
            unsafe {
                cblas_sys::$comp_blas_fn(
                    LEN as i32,
                    a.as_ptr() as *const [$float; 2],
                    1,
                    b.as_ptr() as *const [$float; 2],
                    1,
                    tmp.as_mut_ptr() as *mut [$float; 2],
                )
            }
            Complex {
                re: tmp[0],
                im: tmp[1],
            }
        }

        impl<'a, const LEN: usize> StaticCowVec<'a, Complex<$float>, LEN> {
            /// Dot product for two complex vectors of same length using blas.
            pub fn dot(&self, other: &Self) -> Complex<$float> {
                $comp_blas_fn(self, other)
            }
        }
    };
}

impl_dot!(f32, cblas_sdot, cblas_cdotu_sub);
impl_dot!(f64, cblas_ddot, cblas_zdotu_sub);

impl<'a, T: NumCast + Copy, const LEN: usize> Deref for StaticCowVec<'a, T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        unsafe {
            match self.is_owned {
                true => &self.data.owned,
                false => self.data.borrowed,
            }
        }
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> DerefMut for StaticCowVec<'a, T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            if self.is_owned {
                return &mut self.data.owned;
            }
            self.is_owned = true;
            self.data.owned = *self.data.borrowed;
            &mut self.data.owned
        }
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> From<[T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: [T; LEN]) -> Self {
        Self {
            data: StaticVectorUnion { owned: s },
            is_owned: true,
        }
    }
}
impl<'a, T: NumCast + Copy, const LEN: usize> From<&'a [T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T; LEN]) -> Self {
        Self {
            data: StaticVectorUnion { borrowed: s },
            is_owned: false,
        }
    }
}
impl<'a, T: NumCast + Copy, const LEN: usize> From<&'a [T]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T]) -> Self {
        assert_eq!(s.len(), LEN);
        Self::from(unsafe { &*(s.as_ptr() as *const [T; LEN]) })
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
