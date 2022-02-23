//! <div align="center">
//!
//! # SLAS
//! *Static Linear Algebra System*
//!
//! [![Crates.io](https://img.shields.io/crates/v/slas?logo=rust)](https://crates.io/crates/slas)
//! [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/slas/Tests?label=tests&logo=github)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)
//! [![Coverage Status](https://coveralls.io/repos/github/unic0rn9k/slas/badge.svg?branch=master)](https://coveralls.io/github/unic0rn9k/slas?branch=master)
//! [![Docs](https://img.shields.io/docsrs/slas/latest?logo=rust)](https://docs.rs/slas/latest/slas/)
//! [![Donate on paypal](https://img.shields.io/badge/paypal-donate-1?logo=paypal&color=blue)](https://www.paypal.com/paypalme/unic0rn9k/5usd)
//!
//! </div>
//!
//! A linear algebra system with a focus on performance, static allocation, statically shaped data and copy-on-write (aka cow) behavior.
//! Safe and fast bindings for blas/blis are also provided out of the box.
//!
//! [What is BLAS?](http://www.netlib.org/blas/)
//!
//! ## Example
//!
//! ```rust
//! use slas::prelude::*;
//! let a = moo![f32: 1, 2, 3.2];
//! let b = moo![f32: 3, 0.4, 5];
//! println!("Dot product of {a:?} and {b:?} is {:?}", a.dot(&b));
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
//! By default slas will choose the backend that is assumed to be the fastest, given the amount of elements in the vector.
//!
//! The `StaticCowVec` dereferences to `StaticVecUnion`, which in turn dereferences to `[T; LEN]`,
//! so any method implemented for `[T;LEN]` can also be called on `StaticCowVec` and `StaticVecUnion`.
//!
//! [More example code here.](https://github.com/unic0rn9k/slas/blob/master/tests/src/main.rs)
//!
//! ## What is a cow and when is it useful?
//! The copy-on-write functionality is inspired by [std::borrow::cow](https://doc.rust-lang.org/std/borrow/enum.Cow.html).
//! The idea is simply that allocations (and time) can be saved, by figuring out when to copy at runtime instead of at compiletime.
//! This can be memory inefficient at times (as an enum takes the size of its largest field + tag size), which is why you can optionally use `StaticVecUnion`s and `StaticVec`s instead.
//! You can call `moo`, `moo_ref` and `mut_moo_ref` on any type that implements `StaticVec` to cast it to a appropriate type for it's use-case, with zero overhead.
//!
//! **moo_ref** returns a `StaticVecRef`, which is just a type alias for a reference to a `StaticVecUnion`.
//! This is most efficient when you know you don't need mutable access or ownership of a vector.
//!
//! **mut_moo_ref** returns a `MutStaticVecRef`.
//! This is a lot like `moo_ref`, but is useful when you want to mutate your data in place (fx if you wan't to normalize a vector).
//! You should only use this if you want mutable access to a vector WITH side effects.
//!
//! **moo** returns a `StaticCowVec` that references `self`. This is useful if you don't know if you need mutable access to you vector and you don't want side effects.
//! If you want to copy data into a `StaticCowVec` then `StaticCowVec::from` is what you need.
//!
//! **moo_owned** will just return a `StaticVecUnion`. This is useful when you really just want a `[T; LEN]`,
//! but you need methods only implemented for a `StaticVecUnion`.
//!
//!  ### Example of cow behavior
//! ```rust
//! use slas::prelude::*;
//!
//! let source: Vec<f32> = vec![1., 2., 3.];
//! let mut v = source.moo();
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
//! This is because `v` was copied when it was mutated.
//!
//! ## Matrix example
//!
//! ```rust
//! use slas::prelude::*;
//! use slas_backend::*;
//!
//! let a = moo![f32: 1..=6].matrix::<Blas, 2, 3>();
//! let b = moo![f32: 1..=6].matrix::<Blas, 3, 2>();
//! let c = a.matrix_mul(&b);
//!
//! assert_eq!(c, [22., 28., 49., 64.]);
//!
//! println!("{a:.0?} * {b:.0?} = {:.0?}", c.matrix::<Blas, 2, 2>());
//! ```
//!
//! Indexing into matricies can be done both with columns and rows first.
//! When indexing with `[usize; 2]` it will take columns first, where as using `m!` will be rows first.
//!
//! ```rust
//! use slas::prelude::*;
//! use slas_backend::*;
//!
//! let a = moo![f32: 1..=6].matrix::<Blas, 2, 3>();
//!
//! assert_eq!(a[[0, 1]], a[m![1, 0]]);
//! ```
//!
//! ## Tensor example
//! At the moment tensors can't do much
//! ```
//! use slas::prelude::*;
//! let t = moo![f32: 0..27].reshape(&[3, 3, 3], slas_backend::Rust);
//! assert_eq!(t[[0, 0, 1]], 9.);
//!
//! let mut s = t.index_slice(1);
//!
//! assert_eq!(s[m![0, 0]], 9.);
//! ```
//! That's pretty much it for now...
//!
//! ## Why not just use ndarray (or alike)?
//! Slas can be faster than ndarray in some specific use cases, like when having to do a lot of allocations, or when using referenced data in vector operations.
//! Besides slas should always be atleast as fast as ndarray, so it can't hurt.
//!
//! Ndarray will always use the backend you choose in your `Cargo.toml`.
//! With slas you can choose a backend in code and even create your own backend that fits your needs.
//!
//! Static allocation and the way slas cow behavior works with the borrow checker,
//! also means that you might catch a lot of bugs at compiletime,
//! where ndarray most of the time will let you get away with pretty much anything.
//! For example taking the dot product of two vectors with different sizes,
//! will cause a panic in ndarray and a compiletime error in slas.
//!
//! ## Installation
//! By default slas will assume you have blis installed on your system.
//! If you want tos choose your own blas provider please set `dependencies.slas.default-features = false` in your `Cargo.toml`,
//! and refer to [blas-src](https://lib.rs/crates/blas-src) for further instructions.
//! Remember to add `extern crate blas_src;` if you use blas-src as a blas provider.
//!
//! On the crates.io version of slas (v0.1.0 and 0.1.1) blis is compiled automatically.
//!
//! For now, if you want to use the newest version of slas, you need to install blis/blas on your system.
//! - On Arch linux [blis-cblas](https://aur.archlinux.org/packages/blis-cblas/) v0.7.0 from the AUR has been tested and works fine.
//! - On Debian you can simply run `apt install libblis-dev`.
//! - On Windows [openblas-src](https://github.com/blas-lapack-rs/openblas-src) has been tested.
//! This mean you will need to disable slas default features,
//! follow the installation instructions in the openblas readme and add `extern crate openblas_src` to your main file.
//!
//! ## Misc
//! - Slas is still in very early days, and is subject to a lot of breaking changes.
//! - [Benchmarks, tests and related](https://github.com/unic0rn9k/slas/tree/master/tests)
//!
//! ## TODO
//! [Progress and todos are on trello!](https://trello.com/b/iSakt16M/slas%F0%9F%8C%BF)

#![allow(incomplete_features)]
#![feature(
    generic_const_exprs,
    portable_simd,
    const_fn_trait_bound,
    const_trait_impl,
    const_ptr_as_ref,
    const_option,
    associated_type_defaults,
    const_mut_refs
)]

pub mod prelude;
pub mod tensor;

pub mod backends;
pub use num;

use std::{
    mem::{size_of, transmute},
    ops::*,
};
#[cfg(feature = "blis-sys")]
extern crate blis_src;
extern crate cblas_sys;
mod traits;
use prelude::*;

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
        unsafe { &*(self.as_ptr() as *const [T; LEN]) }
    }

    pub const unsafe fn transmute_elements<U: Copy>(&'a self) -> &'a StaticVecUnion<'a, U, LEN> {
        if size_of::<T>() != size_of::<U>() {
            panic!("Cannot transmute between vectors of different sizes")
        }
        transmute(self)
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
    pub const fn len(&self) -> usize {
        LEN
    }
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn is_borrowed(&self) -> bool {
        !self.is_owned()
    }

    pub const fn is_owned(&self) -> bool {
        self.is_owned
    }

    /// Cast StaticCowVec from pointer.
    ///
    /// # Safety
    /// Is safe as long as `*ptr` is contiguous and `*ptr` has a length of `LEN`.
    pub const unsafe fn from_ptr(ptr: *const T) -> Self {
        Self::from(
            (ptr as *const [T; LEN])
                .as_ref()
                .expect("Cannot create StaticCowVec from null pointer"),
        )
    }

    /// Cast StaticCowVec from pointer without checking if it is null.
    /// **Very** **very** **very** unsafe.
    ///
    /// # Safety
    /// Is safe as long as `*ptr` is contiguous, `*ptr` has a length of `LEN` and `ptr` is not NULL.
    pub const unsafe fn from_ptr_unchecked(ptr: *const T) -> Self {
        Self::from(&*(ptr as *const [T; LEN]))
    }
}

impl<'a, T: Copy, const LEN: usize> const Deref for StaticVecUnion<'a, T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        unsafe { transmute::<&Self, &'a Self::Target>(self) }
    }
}

impl<'a, T: Copy, const LEN: usize> const DerefMut for StaticVecUnion<'a, T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { transmute::<&mut Self, &'a mut Self::Target>(self) }
    }
}

impl<'a, T: Copy, const LEN: usize> const Deref for StaticCowVec<'a, T, LEN> {
    type Target = StaticVecUnion<'a, T, LEN>;

    fn deref(&self) -> &Self::Target {
        if self.is_owned {
            &self.data
        } else {
            unsafe { transmute(self.data.borrowed) }
        }
    }
}

impl<'a, T: Copy, const LEN: usize> const DerefMut for StaticCowVec<'a, T, LEN> {
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

impl<'a, T: Copy, const LEN: usize> const From<[T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: [T; LEN]) -> Self {
        Self {
            data: StaticVecUnion { owned: s },
            is_owned: true,
        }
    }
}
impl<'a, T: Copy, const LEN: usize> const From<&'a [T; LEN]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T; LEN]) -> Self {
        Self {
            data: StaticVecUnion { borrowed: s },
            is_owned: false,
        }
    }
}
impl<'a, T: Copy, const LEN: usize> const From<&'a [T]> for StaticCowVec<'a, T, LEN> {
    fn from(s: &'a [T]) -> Self {
        if s.len() != LEN {
            panic!("Cannot convert slice of incorrect length to StaticCowVec")
        }
        Self::from(unsafe { &*(s.as_ptr() as *const [T; LEN]) })
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

/// Macro for creating [`StaticCowVec`]s
///
/// ## Example
/// ```rust
/// use slas::prelude::*;
/// moo![f32: 1, 2, 3.5];
/// moo![f32: 1..4];
/// moo![f32: 1..=3];
/// moo![0f32; 4];
/// moo![|n|-> f32 { (n as f32).sin() }; 100];
/// moo![|n| (n as f32).sin(); 100];
/// ```
#[macro_export]
macro_rules! moo {
    (|$n: ident| -> $t: ty $do: block ; $len: expr) => {{
        let mut tmp = StaticCowVec::<$t, $len>::from([num!(0); $len]);
        (0..$len).map(|$n| -> f32 {$do}).enumerate().for_each(|(n, v)| tmp[n]=v);
        tmp
    }};
    (|$n: ident| $do: expr ; $len: expr) => {{
        moo![|$n| -> _ {$do}; $len]
    }};
    (on $backend:ty : $($v: tt)*) => {{
        moo![$($v)*].static_backend::<$backend>()
    }};
    (_ $($v: tt)*) => {{
        StaticCowVec::from($($v)*)
    }};
    ($t: ty: $a: literal .. $b: literal) => {{
        let mut tmp = StaticCowVec::from([num!(0); $b - $a]);
        tmp.iter_mut().zip($a..$b).for_each(|(o, i)| *o = i as $t);
        tmp
    }};
    ($t: ty: $a: literal ..= $b: literal) => {{
        let mut tmp = StaticCowVec::from([num!(0); $b - $a+1]);
        tmp.iter_mut().zip($a..=$b).for_each(|(o, i)| *o = i as $t);
        tmp
    }};
    ($t: ty: $($v: expr),* $(,)?) => {{
        StaticCowVec::from([$( $v as $t ),*])
    }};
    ($($v: tt)*) => {{
        StaticCowVec::from([$($v)*])
    }};
}
