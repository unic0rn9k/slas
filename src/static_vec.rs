use crate::prelude::*;
use crate::StaticVecUnion;
use paste::paste;
use std::{
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::DerefMut,
};

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

macro_rules! impl_reshape_unchecked_ref {
	($($mut: tt)?) => {
        paste!{
		    unsafe fn [<reshape_unchecked_ref $(_$mut)?>]<
                'a,
                B: crate::backends::Backend<T>,
                S: crate::tensor::Shape<NDIM>,
                const NDIM: usize,
            >(
                &'a $($mut)? self,
                shape: S,
                backend: B,
            ) -> crate::tensor::Tensor<T, & $($mut)? [T; LEN], B, NDIM, LEN, S>
            where
                Self: Sized,
            {
                Tensor {
                    data: crate::backends::WithStaticBackend::from_static_vec(
                        transmute(self.[< as $(_$mut)? _ptr>]()),
                        backend,
                    ),
                    shape,
                }
            }
        }
	};
}

macro_rules! impl_reshape {
    ($($pub: ident $name_suffix: ident $($t:tt)*)?) => {paste!{
        /// Return [`crate::tensor::Tensor`] with shape [`crate::tensor::MatrixShape::<M, K>`].
        $($pub)? fn [<matrix $($name_suffix)?>]<B: crate::backends::Backend<T>, const M: usize, const K: usize>(
            $($($t)*)? self,
        ) -> crate::tensor::Matrix<T, $($($t)*)? Self, B, LEN, false, MatrixShape<M, K>>
        where
            Self: Sized,
        {
            assert_eq!(
                M * K,
                LEN,
                "Cannot reshape vector of {} elements as matrix of {}",
                LEN,
                M * K
            );
            crate::tensor::Tensor {
                data: crate::backends::WithStaticBackend::from_static_vec(self, B::default()),
                shape: crate::tensor::MatrixShape::<M, K>,
            }
            .into()
        }

        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        ///
        /// let a = moo![f32: 0..6].reshape(&[3, 2], slas_backend::Blas);
        /// let b = [0.; 6].reshape(&[2, 3], slas_backend::Blas);
        ///
        /// assert_eq!(a.matrix().matrix_mul(&b.matrix()), [0.; 4]);
        /// ```
        /// In this example the matricies `a` and `b` have dynamic shapes.
        /// If you wan't to create matricies with static shapes, you should use [`StaticVec::matrix`].
        $($pub)? fn [<reshape $($name_suffix)?>]<
            B: crate::backends::Backend<T>,
            S: crate::tensor::Shape<NDIM>,
            const NDIM: usize,
        >(
            $($($t)*)? self,
            shape: S,
            backend: B,
        ) -> crate::tensor::Tensor<T, $($($t)*)? Self, B, NDIM, LEN, S>
        where
            Self: Sized,
        {
            assert_eq!(
                shape.volume(),
                LEN,
                "Cannot reshape vector with lenght {} as {:?}",
                LEN,
                shape.slice()
            );
            Tensor {
                data: crate::backends::WithStaticBackend::from_static_vec(self, backend),
                shape,
            }
        }
    }};
}

/// Trait for statically shaped, contiguous vectors.
pub trait StaticVec<T, const LEN: usize> {
    /// Return pointer to first element.
    ///
    /// # Safety
    /// Is safe as long as `self` is contiguous.
    unsafe fn as_ptr(&self) -> *const T;

    /// Return mutable pointer to first element.
    ///
    /// # Safety
    /// Is safe as long as `self` is contiguous.
    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        transmute(self.as_ptr())
    }

    /// Return a reference to self with the type of [`StaticVecUnion`]
    fn moo_ref<'a>(&'a self) -> StaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { &*(self.as_ptr() as *const StaticVecUnion<T, LEN>) }
    }

    /// Return a mutable reference to self with the type of [`StaticVecUnion`].
    /// If you want to write to a StaticVec, this is the method that should be used.
    /// This method is re-implemented for StaticCowVecs,
    /// so it perserves cow behavior even when cows are borrowed as StaticVec's.
    fn mut_moo_ref<'a>(&'a mut self) -> MutStaticVecRef<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { &mut *(self.as_mut_ptr() as *mut StaticVecUnion<T, LEN>) }
    }

    /// Return a cow vector containing a reference to self.
    fn moo<'a>(&'a self) -> StaticCowVec<'a, T, LEN>
    where
        T: Copy,
    {
        unsafe { StaticCowVec::from_ptr(self.as_ptr()) }
    }

    /// Indexing without bounds checking.
    ///
    /// # Safety
    /// is safe as long as `i < self.len()`
    unsafe fn get_unchecked<'a>(&'a self, i: usize) -> &'a T {
        &*self.as_ptr().add(i)
    }

    /// Same as [`Self::get_unchecked`] but mutable.
    ///
    /// # Safety
    /// is safe as long as `i < self.len()`
    unsafe fn get_unchecked_mut<'a>(&'a mut self, i: usize) -> &'a mut T
    where
        T: Copy,
    {
        &mut *self.as_mut_ptr().add(i)
    }

    /// Returns a static slice spanning from index i to i+SLEN.
    ///
    /// # Safety
    /// is safe as long as `i+SLEN < self.len()`
    unsafe fn static_slice_unchecked<'a, const SLEN: usize>(&'a self, i: usize) -> &'a [T; SLEN] {
        &*(self.as_ptr().add(i) as *const [T; SLEN])
    }

    /// Returns a mutable static slice spanning from index i to i+SLEN.
    ///
    /// # Safety
    /// is safe as long as `i+SLEN < self.len()`
    unsafe fn mut_static_slice_unchecked<'a, const SLEN: usize>(
        &'a mut self,
        i: usize,
    ) -> &'a mut [T; SLEN] {
        &mut *(self.as_ptr().add(i) as *mut [T; SLEN])
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

    impl_reshape!();
    impl_reshape_unchecked_ref!(mut);
    impl_reshape_unchecked_ref!();
}

impl<'a, T: Copy, const LEN: usize> StaticVecUnion<'a, T, LEN> {
    impl_reshape!(pub _ref &'a);
    impl_reshape!(pub _mut_ref &'a mut);
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

macro_rules! impl_vec_for_refs {
	($($t: tt)*) => {
		impl<T, const LEN: usize> StaticVec<T, LEN> for $($t)* [T; LEN] {
            unsafe fn as_ptr(&self) -> *const T {
                (**self).as_ptr()
            }
            unsafe fn as_mut_ptr(&mut self) -> *mut T {
                if stringify!($($t)*).contains("mut"){
                    (*self).as_mut_ptr()
                }else{
                    panic!("Cannot get mutable pointer from &[T; LEN]. Maybe try &mut [T; LEN] instead.")
                }
            }
        }

        impl<'a, T: Copy, const LEN: usize> StaticVec<T, LEN> for $($t)* StaticVecUnion<'a, T, LEN> {
            unsafe fn as_ptr(&self) -> *const T {
                (**self).as_ptr()
            }
            unsafe fn as_mut_ptr(&mut self) -> *mut T {
                if stringify!($($t)*).contains("mut"){
                    (*self).as_mut_ptr()
                }else{
                    panic!("Cannot get mutable pointer from StaticVecRef<'a, T, LEN>. Maybe try MutStaticVecRef<'a, T, LEN> instead.")
                }
            }
        }

        impl<'a, T: Copy, const LEN: usize> StaticVec<T, LEN> for $($t)* StaticCowVec<'a, T, LEN> {
            unsafe fn as_ptr(&self) -> *const T {
                (**self).as_ptr()
            }
            unsafe fn as_mut_ptr(&mut self) -> *mut T {
                if stringify!($($t)*).contains("mut"){
                    (*self).as_mut_ptr()
                }else{
                    panic!("Cannot get mutable pointer from &StaticCowVec<'a, T, LEN>. Maybe try &mut StaticCowVec<'a, T, LEN> instead.")
                }
            }
        }
	};
}

impl_vec_for_refs!(&);
impl_vec_for_refs!(&mut);
impl_vec_for_refs!(&&);
impl_vec_for_refs!(&mut &mut);

impl<'a, T: Copy, const LEN: usize> StaticVec<T, LEN> for StaticCowVec<'a, T, LEN> {
    unsafe fn as_ptr(&self) -> *const T {
        if self.is_owned {
            self.data.as_ptr()
        } else {
            self.data.borrowed as *const T
        }
    }

    /// For [`StaticCowVec`] calling `as_mut_ptr` will dereference self and thereby copy the contents of self.borrowed into self, if self is borrowed.
    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        if self.is_owned {
            self.data.as_mut_ptr()
        } else {
            transmute(self.mut_moo_ref())
        }
    }

    /// For [`StaticCowVec`] calling `mut_moo_ref` will dereference self and thereby copy the contents of self.borrowed into self, if self is borrowed.
    fn mut_moo_ref<'b>(&'b mut self) -> MutStaticVecRef<'b, T, LEN>
    where
        T: Copy,
    {
        unsafe { transmute(self.deref_mut()) }
    }
}
