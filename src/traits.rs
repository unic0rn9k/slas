use crate::prelude::*;
use crate::StaticVecUnion;
use num::NumCast;
use std::marker::PhantomData;
use std::mem::transmute;

/// A very general trait for anything that can be called a static vector (fx. `[T; LEN]`)
///
/// **Warning:** If self is not contiguous, it will cause undefined behaviour.
///
/// Why does StaticVector not allow mutable access to self?
///
/// Because there is no overhead casting to [`StaticVecUnion::owned`] and calling methods on that instead.
pub trait StaticVec<T, const LEN: usize> {
    /// Return pointer to first element.
    unsafe fn as_ptr(&self) -> *const T;

    /// Return a reference to self with the type of [`StaticVecUnion::owned`]
    fn moo_ref<'a>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { transmute(&self) }
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
/// See [`StaticVec`] for more information.
///
/// ## Example
/// ```rust
/// use slas::prelude::*;
///
/// let a = vec![0f32, 1., 2., 3.];
/// let b = moo![f32: 0, -1, -2, 3];
///
/// assert!(cblas_sdot(&a.pretend_static(), &b) == 4.)
/// ```
pub trait DynamicVec<T> {
    fn len(&self) -> usize;
    unsafe fn as_ptr(&self) -> *const T;
    fn pretend_static<const LEN: usize>(&self) -> PretendStaticVec<'_, T, Self, LEN> {
        dyn_cast_panic!(self.len(), LEN);
        PretendStaticVec(self, PhantomData)
    }
    unsafe fn pretend_static_unchecked<const LEN: usize>(
        &self,
    ) -> PretendStaticVec<'_, T, Self, LEN> {
        PretendStaticVec(self, PhantomData)
    }

    /// Return a reference to self with the type of [`StaticVecUnion::owned`]
    fn moo_ref<'a, const LEN: usize>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { transmute(&self) }
    }

    fn moo<'a, const LEN: usize>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy + NumCast,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }
}

impl<'a, T: Copy, const LEN: usize> StaticVec<T, LEN> for StaticVecUnion<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        self.owned.as_ptr()
    }
}

impl<T, const LEN: usize> StaticVec<T, LEN> for [T; LEN] {
    unsafe fn as_ptr(&self) -> *const T {
        self as *const T
    }
}

/// See [`StaticVec`].
pub struct PretendStaticVec<'a, I, T: DynamicVec<I> + ?Sized, const LEN: usize>(
    &'a T,
    PhantomData<I>,
);

impl<'a, I, T: DynamicVec<I>, const LEN: usize> StaticVec<I, LEN>
    for PretendStaticVec<'a, I, T, LEN>
{
    unsafe fn as_ptr(&self) -> *const I {
        self.0.as_ptr()
    }
}

impl<T> DynamicVec<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn as_ptr(&self) -> *const T {
        self as *const [T] as *const T
    }
}

impl<T> DynamicVec<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn as_ptr(&self) -> *const T {
        &self[0] as *const T
    }
}

impl<'a, T: NumCast + Copy, const LEN: usize> StaticVec<T, LEN> for StaticCowVec<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        match self.is_owned {
            true => self.data.as_ptr(),
            false => self.data.borrowed as *const T,
        }
    }
}
