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
//! [What is BLAS?](http://www.netlib.org/blas/)
//!
//! ## Example
//!
//! ```rust
//! use slas::prelude::*;
//! let a = moo![f32: 1, 2, 3.2];
//! let b = moo![f32: 3, 0.4, 5];
//! println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
//! ```
//! You can also choose a static backend yourself
//! ```rust
//! use slas::prelude::*;
//! let a = moo![on slas_backend::Rust:f32: 1, 2, 3.2];
//! // This will only use rust code for all operations on a
//! ```
//!
//! ```rust
//! use slas::prelude::*;
//! let a = moo![on slas_backend::Blas:f32: 1, 2, 3.2];
//! // This will always use blas for all operations on a
//! ```
//!
//! [More example code here.](https://github.com/unic0rn9k/slas/blob/master/tests/src/main.rs)
//!
//!
//! ## What is a COW?
//! The copy-on-write functionality is inspired by [std::borrow::cow](https://doc.rust-lang.org/std/borrow/enum.Cow.html).
//! The idea is simply that allocations (and time) can be saved, by figuring out when to copy at runtime instead of at compiletime.
//! This can be memory inefficient at times (as an enum takes the size of its largest field + tag size), but I'm planing on making ways around this in the future.
//!
//! **NOTICE:** If you're using the git version of slas, you can now use `StaticVecRef`'s instead of `StaticCowVecs`, when you don't want the cow behavior.
//!
//!  ### In code...
//! ```rust
//! use slas::prelude::*;
//!
//! let source: Vec<f32> = vec![1., 2., 3.];
//! let mut v = moo![_ source.as_slice()];
//!
//! // Here we mutate v,
//! // so the content of source will be copied into v before the mutation occours.
//! v[0] = 0.;
//!
//! assert_eq!(**v, [0., 2., 3.]);
//! assert_eq!(source, vec![1., 2., 3.]);
//! ```
//!
//! The borrow checker won't allow mutating `source` after `v` is created, because assignment to borrowed values is not allowed.
//! This can be a problem in some situations.
//!
//! ```rust
//! use slas::prelude::*;
//!
//! let mut source: Vec<f32> = vec![1., 2., 3.];
//! let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };
//!
//! // Here we can mutate source, because v was created from a raw pointer.
//! source[1] = 3.;
//! v[0] = 0.;
//! source[2] = 4.;
//!
//! assert_eq!(**v, [0., 3., 3.]);
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
//! ].matrix();
//!
//! assert!(m[[1, 0]] == 2.);
//!
//! //let k: Matrix<f32, 3, 2> = moo![f32: 0..6].into();
//!
//! //println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
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
//! By default slas will assume you have blis installed on your system.
//! If you want tos choose your own blas provider please set `dependencies.slas.default-features = false` in your `Cargo.toml`,
//! and refer to [blas-src](https://lib.rs/crates/blas-src) for further instructions.
//! Remember to add `extern crate blas_src;` if you use blas-src as a blas provider.
//!
//! On the crates.io version of slas (v0.1.0 and 0.1.1) blis is compiled automatically.
//!
//! For now, if you want to use the git version of slas, you need to install blis on your system.
//! - On Arch linux `blis-cblas` v0.7.0 from the AUR has been tested and works fine.
//! - On Debian you can simply run `apt install libblis-dev`.
//!
//! ## General info...
//! - Slas is still in very early days, and is subject to a lot of breaking changes.
//! - The `StaticCowVec` type implements `deref` and `deref_mut`, so any method implemented for `[T;LEN]` is also implemented for `StaticCowVec`.
//! - [Benchmarks, tests and related](https://github.com/unic0rn9k/slas/tree/master/tests)
//!
//! ## TODO
//! - ~~Rust version of blas functions allowing for loop unrolling - also compile time checks for choosing fastest function~~
//! - Feature support for conversion between [ndarray](lib.rs/ndarray) types
//! - Allow for use on stable channel - perhabs with a stable feature
//! - Implement stable tensors - perhabs for predefined dimensions with a macro
//! - ~~Make StaticCowVec backed by a union - so that vectors that are always owned can also be supported (useful for memory critical systems, fx. embeded devices).~~
//! - ~~Modular backends - [like in coaster](https://github.com/spearow/juice/tree/master/coaster)~~
//!     - GPU support - maybe with cublas
//!     - ~~Pure rust support - usefull for irust and jupyter support.~~
//!     - `DynacmicBackend` for selecting backends at runtime
//! - ~~Refactor backends to make it more generic~~
//!     - Default backend for default operations
//!
//! ## TODO Before v0.2.0
//! - ~~Feature flag for choosing own blas provider~~
//! - More operations implemented for backends
//! - Rewrite documentation
//! - ~~`WithStaticBackend` struct for vectors with associated backends~~
//! - Write unit tests to make sure unsafe functions can't produce ub.
//! - Make less terrible benchmarks
//! - ~~`Normalize` operation for backends - to prove mutable access to vectors also work in backends, even with StaticCowVecs.~~

#![allow(incomplete_features)]
#![feature(generic_const_exprs, portable_simd)]

mod matrix_stable;
pub use matrix_stable::matrix;
pub mod prelude;

pub mod backends;
pub mod num;

use std::{mem::transmute, ops::*};
#[cfg(feature = "blis-sys")]
extern crate blis_src;
extern crate cblas_sys;
mod traits;
use traits::*;

/// StaticVectorUnion is always owned when it is not found in a StaticCowVec, therefore we have this type alias to make it less confisung when dealing with references to owned vectors.
pub type StaticVecRef<'a, T, const LEN: usize> = &'a StaticVecUnion<'a, T, LEN>;

/// Same as [`StaticVecRef`], but mutable.
pub type MutStaticVecRef<'a, T, const LEN: usize> = &'a mut StaticVecUnion<'a, T, LEN>;

/// Will always be owned, unless inside a [`StaticCowVec`]
#[derive(Clone, Copy, Eq)]
pub union StaticVecUnion<'a, T: Copy, const LEN: usize> {
    owned: [T; LEN],
    borrowed: &'a [T; LEN],
}

impl<'a, T: Copy, const LEN: usize> StaticVecUnion<'a, T, LEN> {
    pub fn slice(&'a self) -> &'a [T; LEN] {
        unsafe { transmute(self.as_ptr()) }
    }
}

impl<'a, T: Copy + PartialEq, const LEN: usize> std::cmp::PartialEq<StaticVecUnion<'a, T, LEN>>
    for StaticVecUnion<'a, T, LEN>
{
    fn eq(&self, other: &Self) -> bool {
        self.slice() == other.slice()
    }
}

/// Vectors as copy-on-write smart pointers. Use full for situations where you don't know, if you need mutable access to your data, at compile time.
/// See [`moo`] for how to create a StaticCowVec.
#[derive(Clone, Copy)]
pub struct StaticCowVec<'a, T: Copy, const LEN: usize> {
    data: StaticVecUnion<'a, T, LEN>,
    is_owned: bool,
}

impl<'a, T: Copy, const LEN: usize> StaticCowVec<'a, T, LEN> {
    pub fn len(&self) -> usize {
        LEN
    }

    pub fn is_borrowed(&self) -> bool {
        !self.is_owned()
    }

    pub fn is_owned(&self) -> bool {
        self.is_owned
    }

    /// Cast StaticCowVec from pointer.
    pub unsafe fn from_ptr(ptr: *const T) -> Self {
        Self::from(
            (ptr as *const [T; LEN])
                .as_ref()
                .expect("Cannot create StaticCowVec from null pointer"),
        )
    }

    /// Cast StaticCowVec from pointer without checking if it is null.
    /// **Very** **very** **very** unsafe.
    pub unsafe fn from_ptr_unchecked(ptr: *const T) -> Self {
        Self::from(transmute::<*const T, &[T; LEN]>(ptr))
    }
}

impl<'a, T: Copy, const LEN: usize> Deref for StaticVecUnion<'a, T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        unsafe { transmute::<&Self, &'a Self::Target>(self) }
    }
}

impl<'a, T: Copy, const LEN: usize> DerefMut for StaticVecUnion<'a, T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { transmute::<&mut Self, &'a mut Self::Target>(self) }
    }
}

impl<'a, T: Copy, const LEN: usize> Deref for StaticCowVec<'a, T, LEN> {
    type Target = StaticVecUnion<'a, T, LEN>;

    fn deref(&self) -> &Self::Target {
        match self.is_owned {
            true => &self.data,
            false => unsafe { transmute(self.data.borrowed) },
        }
    }
}

impl<'a, T: Copy, const LEN: usize> DerefMut for StaticCowVec<'a, T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            if self.is_owned {
                return &mut self.data;
            }
            self.is_owned = true;
            self.data.owned = *self.data.borrowed;
            &mut self.data
        }
    }
}

impl<'a, T: Copy, const LEN: usize> From<[T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: [T; LEN]) -> Self {
        Self {
            data: StaticVecUnion { owned: s },
            is_owned: true,
        }
    }
}
impl<'a, T: Copy, const LEN: usize> From<&'a [T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T; LEN]) -> Self {
        Self {
            data: StaticVecUnion { borrowed: s },
            is_owned: false,
        }
    }
}
impl<'a, T: Copy, const LEN: usize> From<&'a [T]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T]) -> Self {
        assert_eq!(s.len(), LEN);
        Self::from(unsafe { transmute::<*const T, &'a [T; LEN]>(s.as_ptr()) })
    }
}

impl<'a, T: Copy + std::fmt::Debug, const LEN: usize> std::fmt::Debug for StaticCowVec<'a, T, LEN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        if self.is_borrowed() {
            f.write_char('&')?;
        }
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: Copy + std::fmt::Debug, const LEN: usize> std::fmt::Debug
    for StaticVecUnion<'a, T, LEN>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.slice().fmt(f)
    }
}

/// Macro for creating [`StaticCowVec`]'s
///
/// ## Example
/// ```rust
/// use slas::prelude::*;
/// moo![f32: 1, 2, 3.5];
/// moo![f32: 1..4];
/// moo![f32: 1..=3];
/// moo![0f32; 4];
// moo![_ source.as_slice()];
// moo![_: 1f32, 2, 3];
/// ```
#[macro_export]
macro_rules! moo {
    (on $backend:ty : $($v: tt)*) => {{
        moo![$($v)*].static_backend::<$backend>()
    }};
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
