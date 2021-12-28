#![allow(dead_code)]
use crate::{Deref, DerefMut, NumCast, StaticCowVec};

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Shape(pub &'static [usize]);

impl Shape {
    pub const fn ndim(&self) -> usize {
        self.0.len()
    }
    pub fn relative<'b>(&self, shape: Shape) -> usize {
        assert_eq!(self.ndim(), shape.ndim());

        let mut offset = 1;
        let mut res = 0;

        (0..self.ndim()).for_each(|n| {
            res += self.0[n] * offset;
            offset *= shape.0[0];
        });

        res
    }
    pub unsafe fn get_unchecked(&self, i: usize) -> &usize {
        self.0.get_unchecked(i)
    }
}

pub const fn area(i: Shape) -> usize {
    let mut n = 0;
    let mut sum = 1;
    while n < i.ndim() {
        sum *= i.0[n];
        n += 1;
    }
    sum
}

#[derive(Clone, Copy)]
pub struct Tensor<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    data: StaticCowVec<'a, T, { area(SHAPE) }>,
}

impl<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape> Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    pub fn zeros() -> Self {
        assert_eq!(SHAPE.ndim(), NDIM);
        Self {
            data: StaticCowVec::zeros(),
        }
    }

    pub fn slice(&self) -> &[T] {
        &self.data[..]
    }
    //pub fn slice_mut(&mut self) -> &mut [T] {
    //    &mut self.data[..]
    //}

    pub fn is_borrowed(&self) -> bool {
        self.data.is_borrowed()
    }
    pub fn is_owned(&self) -> bool {
        self.data.is_owned()
    }
}

impl<'a, T: NumCast + Copy, const SHAPE: Shape, const NDIM: usize> Deref
    for Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    type Target = StaticCowVec<'a, T, { area(SHAPE) }>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: NumCast + Copy, const SHAPE: Shape, const NDIM: usize> DerefMut
    for Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape> std::ops::Index<[usize; NDIM]>
    for Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    type Output = T;
    fn index(&self, n: [usize; NDIM]) -> &T {
        unsafe {
            assert!(NDIM > 0 && n.get_unchecked(0) < SHAPE.get_unchecked(0));
            let mut i = *n.get_unchecked(0);
            let mut multiplier = 1;

            for j in 1..NDIM {
                assert!(n.get_unchecked(j) < SHAPE.get_unchecked(j));

                multiplier *= SHAPE.get_unchecked(j - 1);
                i += n.get_unchecked(j) * multiplier;
            }
            &self.data[i]
        }
    }
}

impl<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape> std::ops::IndexMut<[usize; NDIM]>
    for Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    fn index_mut(&mut self, n: [usize; NDIM]) -> &mut T {
        unsafe {
            assert!(NDIM > 0 && n.get_unchecked(0) < SHAPE.get_unchecked(0));
            let mut i = *n.get_unchecked(0);
            let mut multiplier = 1;

            for j in 1..NDIM {
                assert!(n.get_unchecked(j) < SHAPE.get_unchecked(j));

                multiplier *= SHAPE.get_unchecked(j - 1);
                i += n.get_unchecked(j) * multiplier;
            }
            &mut self.data[i]
        }
    }
}

impl<'a, T: Copy + NumCast, const NDIM: usize, const SHAPE: Shape>
    From<StaticCowVec<'a, T, { area(SHAPE) }>> for Tensor<'a, T, NDIM, SHAPE>
{
    fn from(a: StaticCowVec<'a, T, { area(SHAPE) }>) -> Self {
        Tensor { data: a }
    }
}

impl<'a, T: Copy + NumCast, const NDIM: usize, const SHAPE: Shape> From<&'a [T; area(SHAPE)]>
    for Tensor<'a, T, NDIM, SHAPE>
{
    fn from(a: &'a [T; area(SHAPE)]) -> Self {
        Tensor {
            data: StaticCowVec::from(a),
        }
    }
}

impl<'a, T: Copy + NumCast, const NDIM: usize, const SHAPE: Shape> From<[T; area(SHAPE)]>
    for Tensor<'a, T, NDIM, SHAPE>
{
    fn from(a: [T; area(SHAPE)]) -> Self {
        Tensor {
            data: StaticCowVec::from(a),
        }
    }
}

#[macro_export]
macro_rules! Tensor {
	[$t: ty : $($s: expr),*] => {
		Tensor::<$t, {[$($s),*].len()}, {Shape(&[$($s),*])}>
	};
}

pub fn zeroooo<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape>(
) -> Tensor<'a, T, NDIM, SHAPE>
where
    StaticCowVec<'a, T, { area(SHAPE) }>: Sized,
{
    assert_eq!(SHAPE.ndim(), NDIM);
    Tensor {
        data: StaticCowVec::zeros(),
    }
}

pub fn sliceeee<'a, T: NumCast + Copy, const NDIM: usize, const SHAPE: Shape>(
    this: &'a Tensor<'a, T, NDIM, SHAPE>,
) -> &'a [T]
where
    [T; area(SHAPE)]: Sized,
{
    match &this.data {
        StaticCowVec::Borrowed(b) => b,
        StaticCowVec::Owned(o) => o,
    }
}

#[cfg(test)]
mod tensors {
    use super::zeroooo;
    use super::Shape;
    use super::Tensor;

    #[test]
    fn zeros() {
        let _: Tensor![f32: 2, 2] = zeroooo();
        //let n: Tensor![f32: 2, 2] = zeroooo();
        //assert!(sliceeee(&m) == n.slice());
    }

    #[test]
    fn mutations() {
        let mut t = <Tensor![f32: 3]>::from(&[0.; 3]);
        assert!(t.data.is_borrowed());
        t[[1]] = 2.;
        assert!(t.data.is_owned());
        (*t)[2] = 1.;
        assert!(**t == [0., 2., 1.])
    }

    //#[test]
    //fn reassignment() {
    //    let mut t = <Tensor![f32: 3]>::from(&[3.; 3]);
    //    assert!(t.data.is_borrowed());
    //    **t = [0.; 3];
    //    assert!(t.data.is_owned());
    //    assert!(**t == [0.; 3]);
    //}

    //#[test]
    //fn tensor_2d() {
    //    let a = [1., 2., 3., 4.];
    //    let m = <Tensor![f32: 2, 2]>::from(&a);
    //    let n = <Tensor![f32: 2, 2]>::from(a);
    //    assert!(**m == **n);
    //}
}
