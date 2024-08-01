#![no_std]
#![doc = include_str!("../README.md")]

use core::ops::{Add, Div, Mul, Neg};
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

/// Extension of [`PolyNum`] for numbers that can be evaluated in a rational polynomial.
///
/// [`One`] and [`PartialOrd`] are required to perform a specific optimization for rational
/// polynomials wherein the input is inverted if the absolute value of the input is greater than 1.
/// This is useful for numerical stability, as it keeps the powers of the input within the range of 0 to 1.
pub trait PolyRational: PolyNum + One + Div<Self, Output = Self> + PartialOrd {}

impl<T> PolyNum for T where
    T: Sized
        + Copy
        + Zero
        + Add<Self, Output = Self>
        + Mul<Self, Output = Self>
        + MulAdd<Self, Self, Output = Self>
{
}

impl<T> PolyRational for T where
    T: PolyNum + One + Neg<Output = Self> + Div<Self, Output = Self> + PartialOrd
{
}

#[inline(always)]
fn fma<F>(x: F, m: F, a: F) -> F
where
    F: MulAdd<F, F, Output = F> + Add<F, Output = F> + Mul<F, Output = F>,
{
    #[cfg(target_feature = "fma")]
    return x.mul_add(m, a);

    #[cfg(not(target_feature = "fma"))]
    return x * m + a;
}

pub mod polynomials;

/// Evaluate a polynomial for an array of coefficients. Can be monomorphized.
///
/// To be monomorphized means a dedicated instance of this code will be generated for
/// the array of this length, removing many/all branches within the internal code that
/// other methods such as [`poly`] may require to support many lengths. This function will
/// be faster, put simply.
#[inline(always)]
pub fn poly_array<F: PolyNum, const N: usize>(x: F, coeffs: &[F; N]) -> F {
    poly_f_n::<F, _, N>(x, |i| unsafe { *coeffs.get_unchecked(i) })
}

/// Evaluate a rational polynomial for an array of coefficients. Can be monomorphized.
///
/// To be monomorphized means a dedicated instance of this code will be generated for
/// the array of this length, removing many/all branches within the internal code that
/// other methods such as [`rational`] may require to support many lengths. This function will
/// be faster, put simply.
#[inline(always)]
pub fn rational_array<F: PolyRational, const N: usize>(
    x: F,
    numerator: &[F; N],
    denomiator: &[F; N],
) -> F {
    rational_f_n::<F, _, _, N>(
        x,
        |i| unsafe { *numerator.get_unchecked(i) },
        |i| unsafe { *denomiator.get_unchecked(i) },
    )
}

/// More flexible variant of [`poly_array`]
#[inline(always)]
pub fn poly_array_t<F: PolyNum, T, const N: usize>(x: F, coeffs: &[T; N]) -> F
where
    T: Clone + Into<F>,
{
    poly_f_n::<F, _, N>(x, |i| unsafe { coeffs.get_unchecked(i).clone().into() })
}

/// More flexible variant of [`rational_array`]
#[inline(always)]
pub fn rational_array_t<F: PolyRational, T, const N: usize>(
    x: F,
    numerator: &[T; N],
    denomiator: &[T; N],
) -> F
where
    T: Clone + Into<F>,
{
    rational_f_n::<F, _, _, N>(
        x,
        |i| unsafe { numerator.get_unchecked(i).clone().into() },
        |i| unsafe { denomiator.get_unchecked(i).clone().into() },
    )
}

/// Evaluate a polynomial for a slice of coefficients. May not be monomorphized.
///
/// To not be monomorphized means this function's codegen may be used for any number of coefficients,
/// and therefore contains branches. It will be faster to use [`poly_array`] instead if possible.
pub fn poly<F: PolyNum>(x: F, coeffs: &[F]) -> F {
    poly_f_internal::<F, _, 0>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

/// Evaluate a rational polynomial for an array of coefficients. May not be monomorphized.
///
/// To not be monomorphized means this function's codegen may be used for any number of coefficients,
/// and therefore contains branches. It will be faster to use [`rational_array`] instead if possible.
///
/// # Panics
///
/// Panics if the numerator and denominator arrays have different lengths.
pub fn rational<F: PolyRational>(x: F, numerator: &[F], denominator: &[F]) -> F {
    assert_eq!(
        numerator.len(),
        denominator.len(),
        "Numerator and Denomiator must have the same number of coefficients"
    );

    rational_f_internal::<F, _, _, 0>(
        x,
        numerator.len(),
        |i| unsafe { *numerator.get_unchecked(i) },
        |i| unsafe { *denominator.get_unchecked(i) },
    )
}

/// Evaluate a polynomial using a function to provide coefficients.
///
/// This function is more flexible than [`poly`] as it allows for the coefficients to be
/// generated on-the-fly. This can be useful for generating coefficients that are not
/// known at compile-time. However, this function may be slower than [`poly`] due to the
/// lack of monomorphization optimizations.
#[inline]
pub fn poly_f<F: PolyNum, G>(x: F, n: usize, g: G) -> F
where
    G: FnMut(usize) -> F,
{
    poly_f_internal::<F, _, 0>(x, n, g)
}

/// Evaluate a rational polynomial using a function to provide coefficients.
///
/// To preserve numerical stability, the rational polynomial is evaluated using the reciprocal of the input
/// if the absolute value of the input `x` is greater than 1. Coefficients will be evaluated
/// in reverse order in this case, forward otherwise. This technique
/// helps keep the powers of `x` in the polynomial within -1 and 1, which is important for
/// numerical stability.
///
/// This function is more flexible than [`rational`] as it allows for the coefficients to be
/// generated on-the-fly. This can be useful for generating coefficients that are not
/// known at compile-time. However, this function may be slower than [`rational`] due to the
/// lack of monomorphization optimizations.
#[inline]
pub fn rational_f<F: PolyRational, N, D>(x: F, n: usize, numerator: N, denomiator: D) -> F
where
    N: FnMut(usize) -> F,
    D: FnMut(usize) -> F,
{
    rational_f_internal::<F, _, _, 0>(x, n, numerator, denomiator)
}

/// Variation of [`poly_f`] that is monomorphized for a specific number of coefficients.
#[inline]
pub fn poly_f_n<F: PolyNum, G, const N: usize>(x: F, g: G) -> F
where
    G: FnMut(usize) -> F,
{
    poly_f_internal::<F, _, N>(x, N, g)
}

/// Variation of [`rational_f`] that is monomorphized for a specific number of coefficients.
///
/// # Panics
///
/// Panics if `L` is 0, as this is not a valid polynomial.
#[inline]
pub fn rational_f_n<F: PolyRational, N, D, const L: usize>(x: F, numerator: N, denomiator: D) -> F
where
    N: FnMut(usize) -> F,
    D: FnMut(usize) -> F,
{
    if L == 0 {
        panic!("Rational polynomials must have at least one coefficient");
    }

    rational_f_internal::<F, _, _, L>(x, L, numerator, denomiator)
}

#[inline(always)]
fn rational_f_internal<F: PolyRational, N, D, const LENGTH: usize>(
    x: F,
    n: usize,
    mut numerator: N,
    mut denominator: D,
) -> F
where
    N: FnMut(usize) -> F,
    D: FnMut(usize) -> F,
{
    let one = F::one();

    // if the length is greater than 2 (degree >= 2) the multiplication will be performed
    // anyway, and LLVM will reuse this result for the non-inverted polynomial below.
    let (numerator, denominator) = if LENGTH > 2 && (x * x) > one {
        let x = one / x;

        (
            poly_f_internal::<_, _, LENGTH>(x, n, |i| numerator(LENGTH - i - 1)),
            poly_f_internal::<_, _, LENGTH>(x, n, |i| denominator(LENGTH - i - 1)),
        )
    } else {
        (
            poly_f_internal::<_, _, LENGTH>(x, n, numerator),
            poly_f_internal::<_, _, LENGTH>(x, n, denominator),
        )
    };

    numerator / denominator
}

#[inline(always)]
#[rustfmt::skip]
fn poly_f_internal<F: PolyNum, G, const LENGTH: usize>(x: F, n: usize, mut g: G) -> F
where
    G: FnMut(usize) -> F,
{
    use polynomials::*;

    // if LENGTH is used, assume n = LENGTH to improve codegen
    if LENGTH > 0 && n != LENGTH {
        unsafe { core::hint::unreachable_unchecked() };
    }

    macro_rules! poly {
        ($name:ident($($pows:expr),*; { $j:expr } + $c:ident[$($coeff:expr),*])) => {{
            $name($($pows,)* $($c($j + $coeff)),*)
        }};
    }

    // fast path for small input
    match n {
        0 => return F::zero(),
        1 => return g(0),
        2 => return poly!(poly_1(x;        {0} + g[0, 1])),
        3 => return poly!(poly_2(x, x * x; {0} + g[0, 1, 2])),
        4 => return poly!(poly_3(x, x * x; {0} + g[0, 1, 2, 3])),
        _ => {}
    }

    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    match n {
        5 =>  return poly!(poly_4 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4])),
        6 =>  return poly!(poly_5 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5])),
        7 =>  return poly!(poly_6 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5, 6])),
        8 =>  return poly!(poly_7 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5, 6, 7])),
        9 =>  return poly!(poly_8 (x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8])),
        10 => return poly!(poly_9 (x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        11 => return poly!(poly_10(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
        12 => return poly!(poly_11(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])),
        13 => return poly!(poly_12(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
        14 => return poly!(poly_13(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])),
        15 => return poly!(poly_14(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])),
        16 => return poly!(poly_15(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])),
        _ => {}
    }

    let x16 = x8 * x8;

    let mut sum = F::zero();

    // Use a hybrid Estrin/Horner algorithm
    let mut j = n;
    while j >= 16 {
        j -= 16;
        sum = fma(sum, x16, poly!(poly_15(x, x2, x4, x8; { j } + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])));
    }

    // handle remaining powers
    let (rmx, res) = match j {
        0  => return sum,
        1  => (x, g(0)),
        2  => (x2,             poly!(poly_1 (x;             {0} + g[0, 1]))),
        3  => (x2*x,           poly!(poly_2 (x, x2;         {0} + g[0, 1, 2]))),
        4  => (x4,             poly!(poly_3 (x, x2;         {0} + g[0, 1, 2, 3]))),
        5  => (x4*x,           poly!(poly_4 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4]))),
        6  => (x4*x2,          poly!(poly_5 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5]))),
        7  => (x4*x2*x,        poly!(poly_6 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5, 6]))),
        8  => (x8,             poly!(poly_7 (x, x2, x4;     {0} + g[0, 1, 2, 3, 4, 5, 6, 7]))),
        9  => (x8*x,           poly!(poly_8 (x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8]))),
        10 => (x8*x2,          poly!(poly_9 (x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))),
        11 => (x8*x2*x,        poly!(poly_10(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))),
        12 => (x8*x4,          poly!(poly_11(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))),
        13 => (x8*x4*x,        poly!(poly_12(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))),
        14 => (x8*x4*x2,       poly!(poly_13(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))),
        15 => ((x8*x4)*(x2*x), poly!(poly_14(x, x2, x4, x8; {0} + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))),
        _  => unsafe { core::hint::unreachable_unchecked() }
    };

    fma(sum, rmx, res)
}
