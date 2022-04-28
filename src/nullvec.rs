use crate::prelude::*;
use paste::*;
use std::marker::PhantomData;

/// A zero sized struct that implements StaticVec.
/// It will always panic when trying to access any data within it.
/// You should generally never have to use this struct,
/// it is only here to avoid performance loss in some cases where functions might expect an argument that ypu know will never be used.
///
/// ```rust
/// use slas::prelude::*;
///
/// unsafe{ NullVec::<f32, 10>::new() };
/// ```
#[derive(Debug)]
pub struct NullVec<T, const LEN: usize>(PhantomData<T>);

impl<T, const LEN: usize> NullVec<T, LEN> {
    pub const unsafe fn new() -> Self {
        Self(PhantomData)
    }
}

macro_rules! impl_null_vec {
    ($({$name: ident, ($($args: expr),*)=> $($body: tt)* }),*) => {
        impl<T, const LEN: usize> StaticVec<T, LEN> for NullVec<T, LEN> {$(
            $($body)* {
                panic!("Tried to call {} on NullVec-{LEN}", stringify!($name))
            }
        )*}

        $(
            paste!{
            #[test]
            #[should_panic]
            #[allow(unused_mut)]
                fn [<$name _test>](){
                    unsafe{
                        let mut a = NullVec::<f32, 10>::new();
                        a.$name($($args)*);
                    }
                }
            }
        )*
    };
}

impl_null_vec! {
    {as_ptr, () => unsafe fn as_ptr(&self) -> *const T},
    {as_mut_ptr, () => unsafe fn as_mut_ptr(&mut self) -> *mut T},
    {moo_ref, () => fn moo_ref<'a>(&'a self) -> StaticVecRef<'a, T, LEN> where T: Copy},
    {mut_moo_ref, () => fn mut_moo_ref<'a>(&'a mut self) -> MutStaticVecRef<'a, T, LEN> where T: Copy},
    {moo, () => fn moo<'a>(&'a self) -> StaticCowVec<'a, T, LEN> where T: Copy},
    {get_unchecked, (0) => unsafe fn get_unchecked<'a>(&'a self, _: usize) -> &'a T},
    {get_unchecked_mut, (0) => unsafe fn get_unchecked_mut<'a>(&'a mut self, _: usize) -> &'a mut T},
    {moo_owned, () => fn moo_owned(&self) -> StaticVecUnion<'static, T, LEN> where T: Copy}
}

#[test]
fn create_null_vec() {
    unsafe { NullVec::<f32, 10>::new() };
}

#[test]
#[should_panic]
fn mutation() {
    let mut a = unsafe { NullVec::<f32, 10>::new() };
    a.mut_moo_ref()[0] = 1.;
}
