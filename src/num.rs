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

macro_rules! unimplemented_complex_ops {
    ($op: tt, $trait: ident, $fn: ident) => {
        impl<T: Float> $trait<Self> for Complex<T> {
            type Output = Self;
            fn $fn(self, _: Self) -> Self {
                unimplemented!()
            }
        }
    };
}

unimplemented_complex_ops!(+, Add, add);
unimplemented_complex_ops!(*, Mul, mul);
unimplemented_complex_ops!(-, Sub, sub);
unimplemented_complex_ops!(/, Div, div);

#[derive(Clone, Copy, Debug, PartialEq)]
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
