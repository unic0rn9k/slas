#![allow(incomplete_features)]
#![feature(test, generic_const_exprs)]

use slas::prelude::*;

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

#[cfg(versus)]
fn main() {
    extern crate test;
    use nalgebra::*;
    use test::black_box;
    let mut a: nalgebra::base::SVector<f32, 3> = [1., 2., 3.1].into();
    let mut b: nalgebra::base::SVector<f32, 3> = [1., -2., 3.].into();

    black_box(slas_backends::Blas.dot(&a, &b));
}

#[cfg(not(versus))]
fn main() {
    extern crate test;
    use test::black_box;

    let a = moo![f32: 1., 2., 3.1];
    let b = moo![f32: 1., -2., 3.];

    black_box(a.dot(&b));
}

#[cfg(test)]
mod thin_blas {
    use crate::*;

    #[test]
    fn casting_and_dot() {
        let a = vec![0f32, 1., 2., 3.];
        let b = moo![f32: 0, -1, -2, 3];

        assert_eq!(slas_backend::Blas.dot(&a.pretend_static(), &b), 4.)
    }

    #[test]
    fn casting_and_dot_alt() {
        use slas::prelude::*;

        let a = vec![0f32, 1., 2., 3.];
        let b = moo![f32: 0, -1, -2, 3];

        assert_eq!(a.moo().dot(&b), 4.);
    }

    #[test]
    fn static_backend() {
        use slas::prelude::*;
        assert_eq!(
            moo![f32: 0..4]
                .static_backend::<slas_backend::Blas>()
                .dot(&[1., 2., 3., 4.].moo_ref().static_backend()),
            20.
        );
    }

    #[test]
    fn static_backend_macro() {
        use slas::prelude::*;
        assert_eq!(
            moo![on slas_backend::Blas:f32: 0..4].dot(&[1., 2., 3., 4.].moo_ref().static_backend()),
            20.
        );

        assert_eq!(
            [1., 2., 3., 4.]
                .moo_ref()
                .static_backend()
                .dot(&moo![on slas_backend::Blas:f32: 0..4]),
            20.
        );
    }
}

#[cfg(test)]
mod numbers {
    #[test]
    fn complex_mul() {
        use slas::prelude::*;
        let a = Complex::<f32> { re: 1., im: 4. };
        let b = Complex::<f32> { re: 5., im: 1. };
        assert_eq!(a * b, Complex { re: 1., im: 21. });
    }

    #[test]
    fn complex_pow() {
        use slas::prelude::*;

        assert_eq!(
            Complex::<f32> { re: 1., im: 4. }.powi_(2),
            Complex { re: -15., im: 8. }
        );

        assert_eq!(
            Complex::<f32> { re: 1., im: 4. }.powi_(0),
            Complex { re: 1., im: 0. }
        );

        assert_eq!(
            Complex::<f32> { re: 1., im: 4. }.powi_(1),
            Complex { re: 1., im: 4. }
        );
    }
}

#[cfg(test)]
mod moo {
    use slas::StaticVecUnion;

    use crate::*;

    #[test]
    fn norm_complex_2d() {
        let c = Complex::<f32> { re: 1.2, im: 2.3 };

        let a = moo![c; 2].static_backend::<slas_backend::Blas>();
        assert_eq!(a.norm(), 3.668787265568828);

        // More accurate on rust, but much slower.
        let a = moo![c; 2].static_backend::<slas_backend::Rust>();
        assert_eq!(a.norm(), 3.668787265568828);
    }

    #[test]
    fn vec_ref_dot() {
        assert_eq!([1f32, 2., 3.,].moo_ref().dot([1., 2., 3.,].moo_ref()), 14.)
    }

    #[test]
    fn dot_slas() {
        assert_eq!(
            slas_backend::Rust.dot(&[1., 2., 3., 4.], &[1., 2., 3., 4.]),
            30.
        );
        assert_eq!(
            slas_backend::Rust.dot(&[1., 2., 3., 4., 5.], &[1., 2., 3., 4., 5.]),
            55.
        );
    }

    #[test]
    fn static_slice_unchecked() {
        unsafe { assert_eq!([1., 2., 3., 4.].static_slice_unchecked::<2>(1), &[2., 3.]) }
    }

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
        assert_eq!(**b, [2., 2., 3.2]);
    }

    #[test]
    fn from_readme() {
        let mut source: Vec<f32> = vec![1., 2., 3.];
        let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };

        // Here we can mutate source, because v was created from a raw pointer.
        source[1] = 3.;
        v[0] = 0.;
        source[2] = 4.;

        assert_eq!(*v, *[0., 3., 3.].moo_ref());
        assert_eq!(source, vec![1., 3., 4.]);
    }

    #[test]
    #[should_panic]
    fn into_is_not_subslice() {
        let _: StaticCowVec<f32, 2> = (&[1., 2., 3.][..]).into();
    }

    #[test]
    fn casting() {
        let _: StaticCowVec<f32, 3> = (&[1., 2., 3.][..]).into();
        let _: StaticCowVec<f32, 3> = (&[1., 2., 3.]).into();
        let _: StaticCowVec<f32, 3> = [1., 2., 3.].into();
    }

    #[test]
    fn transmute_elements() {
        use slas::prelude::*;

        let a = moo![f32: 0..3];
        unsafe {
            let b: &StaticVecUnion<fast_floats::FF32, 3> = a.transmute_elements();
            assert_eq!(a[0], *b[0]);
        }
    }
}

#[cfg(test)]
mod tensors {
    #[test]
    fn matrix() {
        use slas::prelude::*;
        use slas_backend::*;
        let a = moo![f32: 1..=6].matrix::<Blas, 2, 3>();
        let b = moo![f32: 1..=6].matrix::<Blas, 3, 2>();

        // TODO: Use buffer here (type StaticVec),
        // and implement multiplication for matricies without a buffer.
        // Also tensor should implement StaticVec.
        let c = a.matrix_mul(&b);

        assert_eq!(c, [22., 28., 49., 64.]);
    }

    #[test]
    fn trans_matrix() {
        use slas::prelude::*;
        use slas_backend::*;

        let m = moo![f32: 1..=6].matrix::<Rust, 2, 3>();
        assert_eq!(m[(0, 1)], m.transpose()[(1, 0)]);
        assert_eq!(m[(0, 2)], m.transpose()[(2, 0)]);
    }

    #[test]
    #[should_panic]
    fn wrong_size() {
        use slas::prelude::*;
        use slas_backend::*;
        let a = moo![f32: 1..6].matrix::<Blas, 2, 3>();
        let b = moo![f32: 1..6].matrix::<Blas, 3, 2>();

        // TODO: Use buffer here (type StaticVec),
        // and implement multiplication for matricies without a buffer.
        // Also tensor should implement StaticVec.
        let c = a.matrix_mul(&b);

        assert_eq!(c, [22., 28., 49., 64.]);
    }

    #[test]
    fn reshape() {
        use slas::prelude::*;
        let _ = [0f32; 4]
            .static_backend::<slas_backend::Blas>()
            .reshape(&[2, 2]);
    }

    #[test]
    #[should_panic]
    fn wrong_index() {
        use slas::prelude::*;
        let t = moo![f32: 0..27].reshape([3, 3, 3], slas_backend::Rust);
        assert_eq!(t[[3, 0, 0]], 9.);
    }

    #[test]
    fn get_row_from_matrix() {
        use slas::prelude::*;

        let t = moo![f32: 0..9]
            .moo_owned()
            .matrix::<slas_backend::Rust, 3, 3>();

        let t = t.index_slice(1);

        assert_eq!(t[[0]], 3.);
    }

    #[test]
    fn sub_tensors() {
        use slas::prelude::*;

        let t = moo![f32: 0..27]
            .moo_owned()
            .reshape(&[3, 3, 3], slas_backend::Rust);
        let t = t.index_slice(1);

        assert_eq!(t[[0, 0]], 9.);
    }

    #[test]
    #[should_panic]
    fn invalid_sub_tensors_mutation() {
        use slas::prelude::*;

        let t = moo![f32: 0..27]
            .moo_owned()
            .reshape(&[3, 3, 3], slas_backend::Rust);
        let mut t = t.index_slice(1);
        t[[0, 0]] = 8.;
        assert_eq!(t[[0, 0]], 8.);
    }

    #[test]
    fn sub_tensors_mutation() {
        use slas::prelude::*;

        let mut t = moo![f32: 0..27]
            .moo_owned()
            .reshape(&[3, 3, 3], slas_backend::Rust);
        let mut t = t.index_slice_mut(1);
        t[[0, 0]] = 8.;
        assert_eq!(t[[0, 0]], 8.);
    }

    #[test]
    #[should_panic]
    fn sub_tensors_index_out_of_bounds() {
        use slas::prelude::*;

        let t = moo![f32: 0..27]
            .moo_owned()
            .reshape(&[3, 3, 3], slas_backend::Rust);
        let t = t.index_slice(3);

        assert_eq!(t[[2, 2]], 9.);
    }

    #[test]
    fn tensor_2d_to_matrix() {
        use slas::prelude::*;

        let a = moo![f32: 0..6]
            .reshape(&[2, 3], slas_backend::Rust)
            .matrix();

        let b = moo![f32: 0..6]
            .static_backend::<slas_backend::Rust>()
            .matrix::<2, 3>();

        assert_eq!(a.vec_ref().slice(), moo![f32: 0..6].slice());
        assert_eq!(b.vec_ref().slice(), moo![f32: 0..6].slice());
    }

    #[test]
    fn shape() {
        use slas::{
            m,
            tensor::{MatrixShape, Shape},
        };
        let s = [1, 2];
        assert_eq!(s.slice(), &[1, 2]);
        assert_eq!(s.axis_len(0), 1);
        assert_eq!(s.axis_len(1), 2);

        let s = m![2, 1];
        assert_eq!(s.slice(), &[1, 2]);
        assert_eq!(s.axis_len(0), 1);
        assert_eq!(s.axis_len(1), 2);
    }
}

#[cfg(all(test, feature = "versus"))]
mod versus {
    extern crate test;
    use lazy_static::*;
    use rand::random;
    const DOT_ARR_LEN: usize = 100;

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

        #[bench]
        fn dot(be: &mut Bencher) {
            let a: nalgebra::base::SVector<f32, { super::DOT_ARR_LEN }> =
                super::RAND_VECS[0].into();
            let b: nalgebra::base::SVector<f32, { super::DOT_ARR_LEN }> =
                super::RAND_VECS[1].into();

            be.iter(|| black_box(a.dot(&b)));
        }
    }

    mod slas {
        use super::test::{black_box, Bencher};
        use slas::prelude::*;

        #[bench]
        fn dot_default(be: &mut Bencher) {
            let a = super::RAND_VECS[0].moo();
            let b = super::RAND_VECS[1].moo();

            be.iter(|| black_box(a.dot(&b)));
        }

        #[bench]
        fn dot_blas(be: &mut Bencher) {
            let a = super::RAND_VECS[0];
            let b = super::RAND_VECS[1];

            be.iter(|| black_box(slas_backend::Blas.dot(&a, &b)));
        }

        #[bench]
        fn dot_rust(be: &mut Bencher) {
            let a = super::RAND_VECS[0];
            let b = super::RAND_VECS[1];

            be.iter(|| black_box(slas_backend::Rust.dot(&a, &b)));
        }
    }
}
