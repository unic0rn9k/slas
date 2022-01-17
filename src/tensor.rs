use crate::{backends::*, prelude::*};
use std::hint::unreachable_unchecked;

pub trait Shape<const NDIM: usize> {
    fn axis_len(&self, n: usize) -> usize;
    fn volume(&self) -> usize {
        (0..NDIM).map(|n| self.axis_len(n)).product()
    }
}

impl<const LEN: usize> Shape<LEN> for [usize; LEN] {
    fn axis_len(&self, n: usize) -> usize {
        self[n]
    }
}

pub struct MatrixShape<const M: usize, const K: usize>;

impl<const M: usize, const K: usize> const Shape<2> for MatrixShape<M, K> {
    fn axis_len(&self, n: usize) -> usize {
        match n {
            0 => K,
            1 => M,
            _ => unsafe { unreachable_unchecked() },
        }
    }
    fn volume(&self) -> usize {
        M * K
    }
}

pub struct Tensor<
    T,
    U: StaticVec<T, LEN> + 'static,
    B: Backend<T>,
    const NDIM: usize,
    const LEN: usize,
> {
    pub data: WithStaticBackend<T, U, B, LEN>,
    pub shape: &'static dyn Shape<NDIM>,
}

impl<T: Float, B: Backend<T>, U: StaticVec<T, LEN> + 'static, const LEN: usize>
    Tensor<T, U, B, 2, LEN>
{
    pub const fn matrix<const M: usize, const K: usize>(
        data: WithStaticBackend<T, U, B, LEN>,
    ) -> Self
    where
        [T; M * K]: Sized,
    {
        Self {
            data,
            shape: &MatrixShape::<M, K>,
        }
    }
}

impl<
        T: Float + std::fmt::Debug,
        B: Backend<T>,
        U: StaticVec<T, LEN> + 'static,
        const LEN: usize,
    > std::fmt::Debug for Tensor<T, U, B, 2, LEN>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[\n")?;
        let m = self.shape.axis_len(1);
        let k = self.shape.axis_len(0);
        debug_assert_eq!(m * k, LEN);

        for n in 0..m {
            f.write_str("   ")?;
            f.debug_list()
                .entries((n * k..(n + 1) * k).map(|n| unsafe { self.data.data.get_unchecked(n) }))
                .finish()?;
            f.write_str(",\n")?;
        }
        f.write_str("]")?;
        Ok(())
    }
}

impl<
        T: Float + Sized,
        U: StaticVec<T, LEN>,
        B: Backend<T> + operations::MatrixMul<T>,
        const LEN: usize,
    > Tensor<T, U, B, 2, LEN>
{
    pub fn matrix_mul<U2: StaticVec<T, LEN2>, const LEN2: usize, const OLEN: usize>(
        &self,
        other: &Tensor<T, U2, B, 2, LEN2>,
    ) -> [T; OLEN] {
        let m = self.shape.axis_len(1);
        let k = self.shape.axis_len(0);
        let n = other.shape.axis_len(0);

        debug_assert_eq!(self.shape.volume(), LEN);
        debug_assert_eq!(other.shape.volume(), LEN2);
        debug_assert_eq!(m * n, OLEN);

        <B as Backend<T>>::matrix_mul(
            &self.data.backend,
            &self.data.data,
            &other.data.data,
            m,
            n,
            k,
        )
    }
}
