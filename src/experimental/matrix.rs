#![allow(dead_code)]
use crate::experimental::tensor::*;

pub const fn matrix_shape<const K: usize, const M: usize>() -> Shape {
    Shape(&[K, M])
}

/// M corresponds with the number of rows, and K is the columns.
pub type Matrix<'a, T, const M: usize, const K: usize> =
    Tensor<'a, T, 2, { matrix_shape::<M, K>() }>;

//#[derive(Clone, Copy)]
//struct Matrix<'a, T: NumCast + Copy, const M: usize, const K: usize>
//where
//    StaticCowVec<'a, T, { area(matrix_shape::<M, K>()) }>: Sized,
//{
//    data: Tensor<'a, T, 2, { matrix_shape::<M, K>() }>,
//}

#[macro_export]
macro_rules! Matrix {
    //($t: tt: $M:expr, $K: expr => $($v: literal),*) => {{
        ($([$($v: literal),*]),*) => {{
            const fn dim<const M: usize, const K: usize>(m:[[bool; K]; M]) -> Shape{
                assert!(m[0].len() == K && m.len() == M);
                Shape(&[K, M])
            }

            Tensor::<f32, 2, { dim([ $([ $({ $v; false }),* ]),* ]) }>
                ::from([ $($($v),*),* ])

    //Tensor::<$t,2, {matrix_shape::<$M, $K>()}>::from([$($v),*])
    }};
}

#[cfg(test)]
mod matrix {
    use super::*;

    #[test]
    fn zeros() {
        let m = Matrix::<f32, 4, 4>::zeros();
        assert!(**m == [0.; 16]);
    }

    #[test]
    fn zeros_2() {
        let m = Matrix::<f32, 2, 2>::zeros();
        assert!(**m == [0.; 4]);
    }

    #[test]
    fn maggots() {
        //let m: Matrix<f32, 2, 2> = [0.; 4].into();  // This does not work at the moment.
        //let m = Matrix::<f32, 2, 2>::from([0.; 4]); // This does. Ideally both should work :/

        #[rustfmt::skip]
        let m = Matrix![
            [1., 2.],
            [3., 4.]
        ];

        assert!(m[[0, 0]] == 1.) //5., 6., 7., 8., 9.]);
    }
    #[test]
    fn mutations() {
        //let mut m = Matrix::<f32, 3, 3>::from([0.; 9]);
        //m[[2, 2]] = 1.;

        //#[rustfmt::skip]
        //let n = Matrix![
        //    [0., 0., 0.],
        //    [0., 0., 0.],
        //    [0., 0., 1.]
        //];

        //assert!(**m == [0., 0., 0., 0., 0., 0., 0., 0., 1.]);
        //assert!(**n == [0., 0., 0., 0., 0., 0., 0., 0., 1.]);
        //assert!(**n == **m);
    }
}
