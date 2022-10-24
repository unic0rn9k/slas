/// A pure rust slas backend with simd support.
#[derive(Default, Clone, Copy)]
pub struct Rust;
use super::*;
use operations::*;
use std::mem::transmute;
use std::simd::Simd;
use std::simd::SimdFloat;

macro_rules! impl_dot {
    ($t: ty) => {
        /// Pure rust implementation of dot product. This is more performant for smaller vectors,
        /// where as the blas (cblas_sdot and cblas_ddot) dot products are faster for larger vectors.
        ///
        /// ## Example
        /// ```rust
        /// use slas::prelude::*;
        /// assert!(slas_backend::Rust.dot(&[1., 2., 3.], &moo![f32: -1, 2, -1]) == 0.);
        /// ```
        impl DotProduct<$t> for Rust {
            fn dot<const LEN: usize>(
                &self,
                a: &impl StaticVec<$t, LEN>,
                b: &impl StaticVec<$t, LEN>,
            ) -> $t {
                const LANES: usize = crate::simd_lanes::max_for_type::<$t>();

                let mut sum = Simd::<$t, LANES>::from_array([0.; LANES]);
                for n in 0..LEN / LANES {
                    sum += unsafe {
                        Simd::from_slice(a.static_slice_unchecked::<LANES>(n * LANES))
                            * Simd::from_slice(b.static_slice_unchecked::<LANES>(n * LANES))
                    }
                }
                let mut sum = sum.reduce_sum();
                for n in LEN - (LEN % LANES)..LEN {
                    sum += unsafe { a.get_unchecked(n) * b.get_unchecked(n) }
                }
                sum
            }
        }
    };
}

macro_rules! impl_basic_op {
    ($op: ident, $fn: ident, $float_op: tt, $op_assign: ident, $($t: ty),*) => {$(
        /// Basic element wise operators are implemented for all vectors on the rust backend.
        /// This means you can call `a.add(&b)` to add two vectors together.
        /// Whis ofcourse also works with `.sub`, `.mul` and `.div`.
        impl $op<$t> for Rust {
            fn $fn<const LEN: usize>(
                &self,
                a: &impl StaticVec<$t, LEN>,
                b: &impl StaticVec<$t, LEN>,
                c: &mut impl StaticVec<$t, LEN>,
            ) -> () {
                const LANES: usize = crate::simd_lanes::max_for_type::<$t>();

                let out_ptr: *mut [$t; LANES] = unsafe{transmute(c.as_mut_ptr())};

                for n in 0..LEN / LANES {
                    unsafe {
                            *out_ptr.add(n) = transmute(
                                Simd::<$t, LANES>::from_slice(a.static_slice_unchecked::<LANES>(n * LANES)) $float_op
                                Simd::<$t, LANES>::from_slice(b.static_slice_unchecked::<LANES>(n * LANES)))
                    }
                }

                for n in LEN - (LEN % LANES)..LEN {
                    unsafe { *c.get_unchecked_mut(n) = *a.get_unchecked(n) $float_op *b.get_unchecked(n) };
                }
            }
        }

        impl<'a, const LEN: usize> StaticVecUnion<'a, $t, LEN> {
            /// Basic element-wise vector operations, implemented automatically with macro.
            #[inline(always)]
            pub fn $fn(&self, other: &Self) -> Self{
                unsafe{
                    let mut buffer: Self = std::mem::MaybeUninit::uninit().assume_init();
                    $op::$fn(&Rust, self, other, &mut buffer);
                    buffer
                }
            }
        }

        paste!{
            impl<'a, const LEN: usize> StaticVecUnion<'a, $t, LEN> {
                /// Basic element-wise vector operations (buffered), implemented automatically with macro.
                #[inline(always)]
                pub fn [<$fn _into>]<'b>(&self, other: &Self, buffer: MutStaticVecRef<'b, $t, LEN>){
                    $op::$fn(&Rust, self, other, buffer);
                }
            }
        }

        paste!{
            #[test]
            fn [< basic_ $fn _ $t >](){
                let a = moo![$t: 1..13];
                let b = a.$fn(&a);
                let mut c = [0.; 12];
                a.[<$fn _into>](&a, c.mut_moo_ref());

                for n in 0..12{
                    assert_eq!(a[n] $float_op a[n], b[n]);
                    assert_eq!(a[n] $float_op a[n], c[n]);
                }
            }
        }
    )*};
}

macro_rules! impl_norm {
    ($t: ty) => {
        impl Normalize<$t> for Rust {
            type NormOutput = $t;
            fn norm<const LEN: usize>(&self, a: &impl StaticVec<$t, LEN>) -> $t {
                //TODO: Use hypot function here. This will require implementing hypot for all float types first.
                a.moo_ref().iter().map(|&n| n * n).sum::<$t>().sqrt_()
            }

            fn normalize<const LEN: usize>(&self, a: &mut impl StaticVec<$t, LEN>) {
                let norm = Normalize::norm(self, a);
                a.mut_moo_ref().iter_mut().for_each(|n| *n /= norm);
            }
        }

        impl Normalize<Complex<$t>> for Rust {
            type NormOutput = $t;
            fn norm<const LEN: usize>(&self, a: &impl StaticVec<Complex<$t>, LEN>) -> $t {
                //TODO: Use hypot function here. This will require implementing hypot for all float types first.
                a.moo_ref()
                    .iter()
                    .map(|n| Simd::<$t, 2>::from_array([n.re.powi_(2), n.im.powi_(2)]))
                    .sum::<Simd<$t, 2>>()
                    .reduce_sum()
                    .sqrt()
            }

            fn normalize<const LEN: usize>(&self, a: &mut impl StaticVec<Complex<$t>, LEN>) {
                let norm = Normalize::norm(self, a);
                a.mut_moo_ref()
                    .iter_mut()
                    .for_each(|n| *n = *n / norm.into());
            }
        }
    };
}

impl<T: Copy> Transpose<T> for Rust {
    fn transpose_inplace<const LEN: usize>(
        &self,
        a: &mut impl StaticVec<T, LEN>,
        columns: usize,
    ) -> () {
        let mut buffer: [T; LEN] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        <Self as Transpose<T>>::transpose(self, a, &mut buffer, columns);
        **(a.mut_moo_ref()) = buffer
    }

    fn transpose<const LEN: usize>(
        &self,
        a: &impl StaticVec<T, LEN>,
        buffer: &mut impl StaticVec<T, LEN>,
        columns: usize,
    ) -> () {
        for column in 0..columns {
            for row in 0..LEN / columns {
                unsafe {
                    *buffer.get_unchecked_mut(columns * row + column) =
                        *a.get_unchecked(LEN / columns * column + row)
                }
            }
        }
    }
}

impl_norm!(f32);
impl_norm!(f64);

impl_dot!(f32);
impl_dot!(f64);

impl_basic_op!(Addition, add, +, add_assign, f32, f64);
impl_basic_op!(Multiplication, mul, *, mul_assign, f32, f64);
impl_basic_op!(Divition, div, /, div_assign, f32, f64);
impl_basic_op!(Subtraction, sub, -, sub_assign, f32, f64);

impl Backend<f32> for Rust {}
impl Backend<f64> for Rust {}
impl Backend<Complex<f32>> for Rust {}
impl Backend<Complex<f64>> for Rust {}
