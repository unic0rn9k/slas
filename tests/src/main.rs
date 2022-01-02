#![allow(incomplete_features)]
#![feature(test, generic_const_exprs)]

use slas::prelude::*;

fn main() {
    extern crate test;
    use slas::prelude::*;
    use test::black_box;

    let a = [1., 2., 3.1];
    let b = [1., -2., 3.];

    black_box(cblas_sdot(&a, &b));
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
mod thin_blas {
    use crate::*;

    #[test]
    fn casting_and_dot() {
        let a = vec![0f32, 1., 2., 3.];
        let b = moo![f32: 0, -1, -2, 3];

        assert_eq!(cblas_sdot(&a.pretend_static(), &b), 4.)
    }

    #[test]
    fn casting_and_dot_alt() {
        use slas::prelude::*;

        let a = vec![0f32, 1., 2., 3.];
        let b = moo![f32: 0, -1, -2, 3];

        assert_eq!(a.moo().dot(&b), 4.);
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

    #[test]
    fn unsafe_mutations() {
        let mut a: Vec<f32> = vec![1., 2., 3.2];
        let mut b = unsafe { StaticCowVec::<f32, 3>::from_ptr(a.as_ptr()) };
        b[0] = 2.;
        a[0] = 3.;
        assert_eq!(a, vec![3., 2., 3.2]);
        assert_eq!(*b, [2., 2., 3.2]);
    }

    #[test]
    fn from_readme() {
        let mut source: Vec<f32> = vec![1., 2., 3.];
        let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };

        // Here we can mutate source, because v was created from a raw pointer.
        source[1] = 3.;
        v[0] = 0.;
        source[2] = 4.;

        assert_eq!(*v, [0., 3., 3.]);
        assert_eq!(source, vec![1., 3., 4.]);
    }
}

#[cfg(test)]
mod matrix {
    use slas::matrix::Matrix;

    #[test]
    fn zero() {
        let m = Matrix::<f32, 2, 2>::zeros();
        let n: Matrix<f32, 2, 2> = [0.; 4].into();
        assert_eq!(m[[0, 0]], 0.);
        assert_eq!(**m, **n)
    }

    #[test]
    fn mul() {
        let m: Matrix<f32, 2, 3> = [1., 2., 3., 4., 5., 6.].into();
        let n: Matrix<f32, 3, 2> = [10., 11., 20., 21., 30., 31.].into();
        let k = [140., 146., 320., 335.];

        assert_eq!(**(m * n), k);
    }

    #[test]
    fn mul2() {
        let m: Matrix<f32, 4, 3> = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.].into();
        let n: Matrix<f32, 3, 2> = [3., 6., 8., 10., 9., 17.].into();
        let k = [46., 77., 106., 176., 166., 275., 226., 374.];
        assert_eq!(**(m * n), k);
    }

    #[test]
    fn from_readme() {
        use slas::prelude::*;
        let m: Matrix<f32, 2, 3> = [1., 2., 3., 4., 5., 6.].into();
        assert_eq!(m[[1, 0]], 2.);
        let k: Matrix<f32, 3, 2> = moo![f32: 0..6].into();

        println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
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

#[cfg(all(test, feature = "versus"))]
mod versus {
    extern crate test;
    use lazy_static::*;
    use rand::random;
    const DOT_ARR_LEN: usize = 300;

    lazy_static! {
        static ref RAND_VECS: [[f32; DOT_ARR_LEN]; 2] = {
            let mut a = [0f32; DOT_ARR_LEN];
            let mut b = [0f32; DOT_ARR_LEN];
            for n in 0..DOT_ARR_LEN {
                a[n] = random::<f32>();
                b[n] = random::<f32>();
            }
            [a, b]
        };
    }

    mod ndarray {
        use super::test::{black_box, Bencher};
        use ndarray::prelude::*;
        use rand::random;

        #[bench]
        fn dot(be: &mut Bencher) {
            let mut a = Array::zeros(super::DOT_ARR_LEN);
            let mut b = Array::zeros(super::DOT_ARR_LEN);
            for n in 0..super::DOT_ARR_LEN {
                a[n] = super::RAND_VECS[0][n];
                b[n] = super::RAND_VECS[1][n];
            }

            be.iter(|| black_box(a.dot(&b)));
        }
    }

    mod nalgebra {
        use super::test::{black_box, Bencher};
        use rand::random;

        #[bench]
        fn dot(be: &mut Bencher) {
            let mut a: nalgebra::base::SVector<f32, { super::DOT_ARR_LEN }> =
                super::RAND_VECS[0].into();
            let mut b: nalgebra::base::SVector<f32, { super::DOT_ARR_LEN }> =
                super::RAND_VECS[1].into();

            be.iter(|| black_box(a.dot(&b)));
        }
    }

    mod slas {
        use super::test::{black_box, Bencher};
        use rand::random;
        use slas::prelude::*;

        #[bench]
        fn dot(be: &mut Bencher) {
            let mut a = super::RAND_VECS[0].moo();
            let mut b = super::RAND_VECS[1].moo();

            be.iter(|| black_box(a.dot(&b)));
        }

        #[bench]
        fn dot_fast(be: &mut Bencher) {
            be.iter(|| black_box(cblas_sdot(&super::RAND_VECS[0], &super::RAND_VECS[1])));
        }
    }
}
