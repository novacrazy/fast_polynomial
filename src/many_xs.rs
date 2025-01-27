use core::ops::{Add, Div, Index, Mul};

use num_traits::{MulAdd, One, Zero};

#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct ArrayWrap<const N: usize, F> {
    underlying: [F; N],
}

impl<const N: usize, F> ArrayWrap<N, F> {
    pub fn new(underlying: [F; N]) -> Self {
        Self { underlying }
    }
}

impl<const N: usize, F> Add for ArrayWrap<N, F>
where
    F: Add<F, Output = F> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let underlying = core::array::from_fn(|idx| self.underlying[idx] + rhs.underlying[idx]);
        Self { underlying }
    }
}

impl<const N: usize, F> Zero for ArrayWrap<N, F>
where
    F: Zero + Add<F, Output = F> + Copy,
{
    fn zero() -> Self {
        Self {
            underlying: [F::zero(); N],
        }
    }

    fn is_zero(&self) -> bool {
        self.underlying.iter().all(Zero::is_zero)
    }
}

impl<const N: usize, F> Mul for ArrayWrap<N, F>
where
    F: Mul<F, Output = F> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let underlying = core::array::from_fn(|idx| self.underlying[idx] * rhs.underlying[idx]);
        Self { underlying }
    }
}

impl<const N: usize, F> MulAdd<Self, Self> for ArrayWrap<N, F>
where
    F: MulAdd<F, Output = F> + Copy,
{
    type Output = Self;

    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        let underlying = core::array::from_fn(|idx| {
            self.underlying[idx].mul_add(a.underlying[idx], b.underlying[idx])
        });
        Self { underlying }
    }
}

impl<const N: usize, F> MulAdd<F, Self> for ArrayWrap<N, F>
where
    F: MulAdd<F, F, Output = F> + Copy,
{
    type Output = Self;

    fn mul_add(self, a: F, b: Self) -> Self::Output {
        let underlying =
            core::array::from_fn(|idx| self.underlying[idx].mul_add(a, b.underlying[idx]));
        Self { underlying }
    }
}

impl<const N: usize, F> Mul<F> for ArrayWrap<N, F>
where
    F: Mul<F, Output = F> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        let underlying = core::array::from_fn(|idx| self.underlying[idx] * rhs);
        Self { underlying }
    }
}

impl<const N: usize, F> From<F> for ArrayWrap<N, F>
where
    F: Copy,
{
    fn from(value: F) -> Self {
        let underlying = [value; N];
        Self { underlying }
    }
}

impl<const N: usize, F> Index<usize> for ArrayWrap<N, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.underlying[index]
    }
}

impl<const N: usize, F: PartialOrd> PartialOrd for ArrayWrap<N, F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.underlying.partial_cmp(&other.underlying)
    }
}

impl<const N: usize, F> One for ArrayWrap<N, F>
where
    F: One + Copy,
{
    fn one() -> Self {
        Self::new([F::one(); N])
    }
}

impl<const N: usize, F> Div for ArrayWrap<N, F>
where
    F: Div<F, Output = F> + Copy,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let underlying = core::array::from_fn(|idx| self.underlying[idx] / rhs.underlying[idx]);
        Self { underlying }
    }
}
