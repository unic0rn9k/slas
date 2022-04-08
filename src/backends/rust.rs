/// A pure rust slas backend with simd support.
#[derive(Default, Clone, Copy)]
pub struct Rust;
use super::*;
use operations::*;
use std::simd::Simd;

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

macro_rules! impl_add {
    ($t: ty) => {
        impl Addition<$t> for Rust {
            fn add<const LEN: usize>(
                &self,
                a: &mut impl StaticVec<$t, LEN>,
                b: &impl StaticVec<$t, LEN>,
            ) -> () {
                const LANES: usize = crate::simd_lanes::max_for_type::<$t>();

                let a = a.mut_moo_ref();

                for n in 0..LEN / LANES {
                    unsafe {
                        *std::mem::transmute::<_, *mut Simd<$t, LANES>>(a.as_mut_ptr())
                            .add(n * LANES) +=
                            Simd::from_slice(a.static_slice_unchecked::<LANES>(n * LANES))
                                + Simd::from_slice(b.static_slice_unchecked::<LANES>(n * LANES))
                    }
                }

                for n in LEN - (LEN % LANES)..LEN {
                    unsafe { *a.get_unchecked_mut(n) += b.get_unchecked(n) };
                }
            }
        }
    };
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
impl_add!(f32);
impl_add!(f64);

impl Backend<f32> for Rust {}
impl Backend<f64> for Rust {}
impl Backend<Complex<f32>> for Rust {}
impl Backend<Complex<f64>> for Rust {}
