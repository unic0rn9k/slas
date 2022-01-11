use crate::prelude::*;
use crate::StaticVecUnion;
use std::marker::PhantomData;
use std::mem::transmute;
use std::mem::transmute_copy;
use std::ops::DerefMut;

/// A very general trait for anything that can be called a static vector (fx. `[T; LEN]`)
///
/// **Warning:** If self is not contiguous, it will cause undefined behaviour.
///
/// Why does StaticVector not allow mutable access to self?
///
/// Because there is no overhead casting to [`StaticVecUnion`] and calling methods on that instead.
///
/// ## Example implementation
/// ```rust
/// use slas::prelude::*;
///
/// struct StaticVec3([f32; 3]);
///
/// impl StaticVec<f32, 3> for StaticVec3{
///     unsafe fn as_ptr(&self) -> *const f32{
///         &self.0[0] as *const f32
///     }
/// }
/// ```

//TODO: Likely need to move deref and deref_mut into this trait, to avoid weird behavior with passing StaticCowVec as &mut impl StaticVec.
/// Trait for statically shaped, contiguous vectors.
pub trait StaticVec<T, const LEN: usize> {
    /// Return pointer to first element.
    unsafe fn as_ptr(&self) -> *const T;

    /// Return a reference to self with the type of [`StaticVecUnion`]
    fn moo_ref<'a>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { transmute(self.as_ptr()) }
    }

    /// Return a mutable reference to self with the type of [`StaticVecUnion`].
    /// If you want to write to a StaticVec, this is the method that should be used.
    /// This method is re-implemented for StaticCowVecs,
    /// so it perserves cow behavior even when cows are borrowed as StaticVec's.
    fn mut_moo_ref<'a>(&'a mut self) -> MutStaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { transmute(self.as_ptr()) }
    }

    /// Return a cow vector containing a reference to self.
    fn moo<'a>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }

    /// Indexing without bounds checking.
    unsafe fn get_unchecked<'a>(&'a self, i: usize) -> &'a T {
        transmute(self.as_ptr().offset(i as isize))
    }

    /// Returns a static slice spanning from index i to i+SLEN.
    unsafe fn static_slice_unchecked<'a, const SLEN: usize>(&'a self, i: usize) -> &'a [T; SLEN] {
        transmute::<*const T, &'a [T; SLEN]>(self.as_ptr().offset(i as isize))
    }

    /// Copies self into a StaticVecUnion.
    fn moo_owned(&self) -> StaticVecUnion<'static, T, LEN>
    where
        T: Copy,
        Self: Sized,
    {
        unsafe { transmute_copy(self) }
    }

    /// Statically use `B` as a backend for self.
    fn static_backend<B: Backend<T> + Default>(
        self,
    ) -> crate::backends::WithStaticBackend<T, Self, B, LEN>
    where
        Self: Sized,
    {
        crate::backends::WithStaticBackend {
            data: self,
            backend: B::default(),
            _pd: PhantomData::<T>,
        }
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
/// assert!(a.moo_ref().dot(b.moo_ref()) == 4.);
/// ```
///
/// ## Example implementation
/// ```rust
/// use slas::prelude::*;
///
/// struct DynVec32(Vec<f32>);
///
/// impl DynamicVec<f32> for DynVec32{
///     fn len(&self) -> usize{
///         self.0.len()
///     }
///
///     unsafe fn as_ptr(&self) -> *const f32{
///         &self.0[0] as *const f32
///     }
/// }
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

    /// Return a reference to self with the type of [`StaticVecUnion`]
    fn moo_ref<'a, const LEN: usize>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { transmute(self.as_ptr()) }
    }

    fn moo<'a, const LEN: usize>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy,
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

/// Pretend dynamically shaped data is statical, meaning it implements [`StaticVec`].
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

impl<'a, T: Copy, const LEN: usize> StaticVec<T, LEN> for StaticCowVec<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        match self.is_owned {
            true => self.data.as_ptr(),
            false => self.data.borrowed as *const T,
        }
    }

    fn mut_moo_ref<'b>(&'b mut self) -> MutStaticVecRef<'b, T, LEN>
    where
        T: Copy,
    {
        unsafe { transmute(self.deref_mut()) }
    }
}
