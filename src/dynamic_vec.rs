use crate::prelude::*;
use std::marker::PhantomData;
use std::mem::transmute;

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
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pointer to first element in a dynamic vector.
    ///
    /// # Safety
    /// is safe as long as `self` is contiguous.
    unsafe fn as_ptr(&self) -> *const T;

    /// Mutable pointer to first element in a dynamic vector.
    ///
    /// # Safety
    /// is safe as long as `self` is contiguous.
    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        transmute(self.as_ptr())
    }

    /// Pretend a dynamic vector is static.
    ///
    /// # Safety
    /// is safe as long as `self` is contiguous.
    /// will panic if `self.len() != LEN`
    fn pretend_static<const LEN: usize>(self) -> PretendStaticVec<T, Self, LEN>
    where
        Self: Clone,
    {
        dyn_cast_panic!(self.len(), LEN);
        PretendStaticVec(Box::new(self), PhantomData)
    }

    /// Pretend a dynamic vector is static without checking if `self.len() == LEN`.
    ///
    /// # Safety
    /// is safe as long as `self.len() == LEN` and `self` is contiguous.
    unsafe fn pretend_static_unchecked<const LEN: usize>(self) -> PretendStaticVec<T, Self, LEN>
    where
        Self: Clone,
    {
        PretendStaticVec(Box::new(self), PhantomData)
    }

    /// Return a reference to self with the type of [`StaticVecUnion`]
    fn moo_ref<'a, const LEN: usize>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { transmute(self.as_ptr()) }
    }

    /// Get a mutable static vector reference from a dynamic vector.
    fn mut_moo_ref<'b, const LEN: usize>(&'b mut self) -> MutStaticVecRef<'b, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { transmute(self.as_mut_ptr()) }
    }

    fn moo<'a, const LEN: usize>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy,
    {
        dyn_cast_panic!(self.len(), LEN);
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }
}

/// Pretend dynamically shaped data is statical, meaning it implements [`StaticVec`].
///
/// ## Example
/// ```rust
/// use slas::prelude::*;
/// moo![f32: 1, 2, 3].dot(vec![1., 2., 3.].pretend_static().moo_ref());
/// ```
pub struct PretendStaticVec<I, T: DynamicVec<I> + ?Sized, const LEN: usize>(Box<T>, PhantomData<I>);

impl<I, T: DynamicVec<I>, const LEN: usize> StaticVec<I, LEN> for PretendStaticVec<I, T, LEN> {
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

impl<T> DynamicVec<T> for Box<[T]> {
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    unsafe fn as_ptr(&self) -> *const T {
        self as *const Box<[T]> as *const T
    }
}
