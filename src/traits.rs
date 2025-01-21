use core::ops::{Add, Div, Mul};
use num_traits::{MulAdd, One, Zero};

/// The minimum required functionality for a number to evaluated in a polynomial. [`MulAdd`]
/// is required to allow for the fused multiply-add operation to be used, which can be
/// faster and more numerically stable than separate multiply and add operations.
///
/// # Note
///
/// For fused multiply-add to be used, the target feature `fma` must be enabled. This can be
/// done by editing your `RUSTFLAGS` environment variable to include `-C target-feature=+fma`,
/// or by editing your `.cargo/config.toml` to include:
///
/// ```toml
/// [build]
/// rustflags = ["-C", "target-feature=+fma"]
/// ```
pub trait PolyNum:
    Sized
    + Copy
    + Zero
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + MulAdd<Self, Self, Output = Self>
{
}

impl<T> PolyNum for T where
    T: Sized
        + Copy
        + Zero
        + Add<Self, Output = Self>
        + Mul<Self, Output = Self>
        + MulAdd<Self, Self, Output = Self>
{
}

#[cfg(feature = "fma")]
pub trait PolyInOut<F>: PolyNum + MulAdd<F, Self, Output = Self> {}

#[cfg(feature = "fma")]
impl<F, T> PolyInOut<F> for T where T: PolyNum + MulAdd<F, Self, Output = Self> {}

#[cfg(not(feature = "fma"))]
pub trait PolyInOut<F>: PolyNum + Mul<F, Output = Self> {}

#[cfg(not(feature = "fma"))]
impl<F, T> PolyInOut<F> for T where T: PolyNum + Mul<F, Output = Self> {}

/// Extension of [`PolyNum`] for numbers that can be evaluated in a rational polynomial.
///
/// [`One`] and [`PartialOrd`] are required to perform a specific optimization for rational
/// polynomials wherein the input is inverted if the absolute value of the input is greater than 1.
/// This is useful for numerical stability, as it keeps the powers of the input within the range of 0 to 1.
pub trait PolyRationalInOut<F>:
    PolyInOut<F> + One + Div<Self, Output = Self> + PartialOrd + From<F>
{
}

impl<T, F> PolyRationalInOut<F> for T where
    T: PolyInOut<F> + One + Div<Self, Output = Self> + PartialOrd + From<F>
{
}
