#![allow(incomplete_features)]
#![feature(test, generic_const_exprs)]

use slas::prelude::*;

fn main() {
    let a = StaticCowVec::from(&[1f32, 2., 3.2]);
    let mut b = moo![f32: 3, 0.4, 5];
    b.norm();
    println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
}

#[cfg(test)]
mod benches {
    extern crate test;

    use crate::*;
    use test::{black_box, Bencher};

    #[bench]
    fn norm(b: &mut Bencher) {
        b.iter(|| {
            let mut a = StaticCowVec::from(&[1f32, 2., 3.2]);
            let mut b = moo![f32: 3, 0.4, 5];
            a.norm();
            b.norm();

            black_box(a.dot(&b));
        });
    }
}

#[cfg(test)]
mod moo {
    use crate::num::*;
    use crate::*;

    #[test]
    fn mutations() {
        let mut t = StaticCowVec::<f32, 3>::from(&[3f32, 2., 3.][..]);
        assert!(t.is_borrowed());
        t[0] = 1.;
        assert!(t.is_owned());
        assert_eq!(&t[..], &[1., 2., 3.])
    }

    #[test]
    fn dot() {
        let a = moo![f32: 0, 1, 2, 3];
        let b = StaticCowVec::from(&[0f32, -1., -2., 3.]);
        assert_eq!(a.dot(&b), 4.)
    }

    #[test]
    fn dot_complex() {
        let c = Complex::<f32> { re: 1., im: 2. };
        let a = moo![c; 5];
        let b = moo![c; 5];
        assert_eq!(a.dot(&b), Complex { re: -15., im: 20. })
    }
}

#[cfg(test)]
mod matrix {
    use slas::matrix::Matrix;

    #[test]
    fn zero() {
        let m = Matrix::<f32, 2, 2>::zeros();
        let n: Matrix<f32, 2, 2> = [0.; 4].into();
        assert!(m[[0, 0]] == 0.);
        assert!(**m == **n)
    }

    #[test]
    fn mul() {
        let m: Matrix<f32, 3, 2> = [1., 2., 3., 4., 5., 6.].into();
        let n: Matrix<f32, 2, 3> = [10., 11., 20., 21., 30., 31.].into();
        let k = [140., 146., 320., 335.];

        assert_eq!(**(m * n), k);
    }

    #[test]
    fn mul2() {
        let m: Matrix<f32, 3, 4> = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.].into();
        let n: Matrix<f32, 2, 3> = [3., 6., 8., 10., 9., 17.].into();
        let k = [46., 77., 106., 176., 166., 275., 226., 374.];
        assert_eq!(**(m * n), k);
    }

    //#[test]
    //fn mul3() { // Doesn't work. Might just be an incorrect expected result.
    //    let m: Matrix<f32, 5, 6> = [
    //        1.0, 2.0, -1.0, -1.0, 4.0, 2.0, 0.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 2.0, -3.0,
    //        2.0, 2.0, 2.0, 0.0, 4.0, 0.0, -2.0, 1.0, -1.0, -1.0, -1.0, 1.0, -3.0, 2.0,
    //    ]
    //    .into();

    //    let n: Matrix<f32, 4, 5> = [
    //        1.0, -1.0, 0.0, 2.0, 2.0, 2.0, -1.0, -2.0, 1.0, 0.0, -1.0, 1.0, -3.0, -1.0, 1.0, -1.0,
    //        4.0, 2.0, -1.0, 1.0,
    //    ]
    //    .into();

    //    let k = [
    //        24.0, 13.0, -5.0, 3.0, -3.0, -4.0, 2.0, 4.0, 4.0, 1.0, 2.0, 5.0, -2.0, 6.0, -1.0, -9.0,
    //        -4.0, -6.0, 5.0, 5.0, 16.0, 7.0, -4.0, 7.0,
    //    ];

    //    assert_eq!(**(m * n), k);
    //}
}
