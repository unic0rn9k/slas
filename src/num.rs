//! This code is mostly just ripped from [num-complex](https://github.com/rust-num/num-complex/blob/3a89daa2c616154035dd27d706bf7938bcbf30a8/src/lib.rs).

use std::ops::*;

pub trait Float:
    Add<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Copy
    + From<f32>
    + PartialEq
{
    fn zero() -> Self {
        Self::from(0f32)
    }
    fn sqrt_(self) -> Self;
    fn powi_(self, p: i32) -> Self;
    fn hypot_(self, other: Self) -> Self;
}

macro_rules! impl_float {
    ($t: ty) => {
        impl Float for $t {
            fn sqrt_(self) -> Self {
                self.sqrt()
            }
            fn powi_(self, p: i32) -> Self {
                self.powi(p)
            }
            fn hypot_(self, other: Self) -> Self {
                self.hypot(other)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

macro_rules! impl_complex_plus_min {
    ($op: tt, $trait: ident, $fn: ident) => {
        impl<T: Float> $trait<Self> for Complex<T> {
            type Output = Self;
            fn $fn(self, other: Self) -> Self {
                Self::Output{re: self.re $op other.re, im: self.im $op other.im}
            }
        }
    };
}

impl_complex_plus_min!(+, Add, add);
impl_complex_plus_min!(-, Sub, sub);

impl<T: Float> Mul<Self> for Complex<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let re = self.re * other.re - self.im * other.im;
        let im = self.re * other.im + self.im * other.re;
        Self::Output { re, im }
    }
}
impl<T: Float> Div<Self> for Complex<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let re = self.re * other.re + self.im * other.im;
        let im = self.re * other.im - self.im * other.re;
        Self::Output { re, im }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Complex<T: Float> {
    pub re: T,
    pub im: T,
}

impl<T: Float> From<T> for Complex<T> {
    fn from(n: T) -> Self {
        Self {
            re: n,
            im: T::zero(),
        }
    }
}

impl<T: Float> Float for Complex<T>
where
    Complex<T>: From<f32>,
{
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    /// This function os currently unimplemented for complex numbers, you can still use `number * number`.
    fn powi_(self, n: i32) -> Self {
        let mut prod = Complex::<T> {
            re: T::from(1.),
            im: T::from(0.),
        };
        for _ in 0..n {
            prod = prod * self
        }
        prod
    }

    fn sqrt_(self) -> Self {
        let norm = self.re.hypot_(self.im);
        Self {
            re: ((norm + self.re) / 2f32.into()).sqrt_(),
            im: ((norm - self.re) / 2f32.into()).sqrt_(),
        }
    }

    fn hypot_(self, other: Self) -> Self {
        (self * self + other * other).sqrt_()
    }
}

impl<T: Float> From<[T; 2]> for Complex<T> {
    fn from(n: [T; 2]) -> Self {
        Complex { re: n[0], im: n[1] }
    }
}

use paste::paste;
macro_rules! gen_tests {
    ($t: ty) => {
        paste! {
            #[cfg(test)]
            #[test]
            fn [<procedural_tests _ $t>]() {
                for x in 0..100 {
                    let x = x as $t / 2.;
                    for y in 0..100 {
                        let y = y as $t / 2.;
                        assert_eq!(x.hypot(y), x.hypot_(y));
                    }
                    for n in 0..4 {
                        assert_eq!(x.powi(n), x.powi_(n));
                    }
                    assert_eq!(x.sqrt(), x.sqrt_());
                }
            }
        }
    };
}

gen_tests!(f32);
gen_tests!(f64);
