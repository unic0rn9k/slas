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
    fn sqrt_(&self) -> Self;
    fn powi_(&self, p: i32) -> Self;
}

macro_rules! impl_float {
    ($t: ty) => {
        impl Float for $t {
            fn sqrt_(&self) -> Self {
                self.sqrt()
            }
            fn powi_(&self, p: i32) -> Self {
                self.powi(p)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

macro_rules! unimplemented_complex_plus_min {
    ($op: tt, $trait: ident, $fn: ident) => {
        impl<T: Float> $trait<Self> for Complex<T> {
            type Output = Self;
            fn $fn(self, other: Self) -> Self {
                Self::Output{re: self.re $op other.re, im: self.im $op other.im}
            }
        }
    };
}

unimplemented_complex_plus_min!(+, Add, add);
unimplemented_complex_plus_min!(-, Sub, sub);

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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Complex<T: Float> {
    pub re: T,
    pub im: T,
}

impl<T: Float> From<f32> for Complex<T> {
    fn from(n: f32) -> Self {
        Self {
            re: n.into(),
            im: T::zero(),
        }
    }
}

impl<T: Float> Float for Complex<T> {
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    /// This function os currently unimplemented for complex numbers, you can still use `number * number`.
    fn powi_(&self, _: i32) -> Self {
        unimplemented!()
    }

    fn sqrt_(&self) -> Self {
        let norm = (self.re.powi_(2) + self.im.powi_(2)).sqrt_();
        Self {
            re: ((norm + self.re) / 2f32.into()).sqrt_(),
            im: ((norm - self.re) / 2f32.into()).sqrt_(),
        }
    }
}
