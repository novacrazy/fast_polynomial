use core::ops::{Add, Div, Mul};
use num_traits::{MulAdd, One, Zero};

/// The minimum required functionality for coefficients of a polynomial. [`MulAdd`]
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
pub trait PolyCoeff:
    Copy
    + Zero
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + MulAdd<Self, Self, Output = Self>
{
}

impl<T> PolyCoeff for T where
    T: Copy
        + Zero
        + Add<Self, Output = Self>
        + Mul<Self, Output = Self>
        + MulAdd<Self, Self, Output = Self>
{
}

#[cfg(feature = "fma")]
/// The minimum required functionality for a number (or many numbers simultaneously)
/// to evaluated in a polynomial with `F` coefficients.
pub trait PolyInOut<F>: PolyCoeff + MulAdd<F, Self, Output = Self> + From<F> {}

#[cfg(feature = "fma")]
impl<F, T> PolyInOut<F> for T where T: PolyCoeff + MulAdd<F, Self, Output = Self> + From<F> {}

#[cfg(not(feature = "fma"))]
/// The minimum required functionality for a number (or many numbers simultaneously)
/// to evaluated in a polynomial with `F` coefficients.
pub trait PolyInOut<F>: PolyCoeff + Mul<F, Output = Self> + From<F> {}

#[cfg(not(feature = "fma"))]
impl<F, T> PolyInOut<F> for T where T: PolyCoeff + Mul<F, Output = Self> + From<F> {}

/// Extension of [`PolyInOut`] for numbers that can be evaluated in a rational polynomial.
///
/// [`One`] and [`PartialOrd`] are required to perform a specific optimization for rational
/// polynomials wherein the input is inverted if the absolute value of the input is greater than 1.
/// This is useful for numerical stability, as it keeps the powers of the input within the range of 0 to 1.
pub trait PolyRationalInOut<F>: PolyInOut<F> + One + Div<Self, Output = Self> + PartialOrd {}

impl<T, F> PolyRationalInOut<F> for T where
    T: PolyInOut<F> + One + Div<Self, Output = Self> + PartialOrd
{
}
