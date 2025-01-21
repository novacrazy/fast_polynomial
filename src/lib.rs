#![no_std]
#![doc = include_str!("../README.md")]
#![allow(clippy::inline_always)]

use traits::{PolyInOut, PolyNum, PolyRationalInOut};

pub mod many_xs;
mod polynomials;
mod traits;

#[inline(always)]
fn fma<F0: PolyInOut<F>, F: PolyNum>(x: F0, m: F, a: F0) -> F0 {
    #[cfg(feature = "fma")]
    return x.mul_add(m, a);
    #[cfg(not(feature = "fma"))]
    return x * m + a;
}

/// Evaluate a polynomial for an array of coefficients. Can be monomorphized.
///
/// To be monomorphized means a dedicated instance of this code will be generated for
/// the array of this length, removing many/all branches within the internal code that
/// other methods such as [`poly`] may require to support many lengths. This function will
/// be faster, put simply.
#[inline]
pub fn poly_array<F0: PolyInOut<F1> + From<F1>, F1: PolyNum, const N: usize>(
    x: F0,
    coeffs: &[F1; N],
) -> F0 {
    poly_f_n::<F0, F1, _, N>(x, |i| unsafe { *coeffs.get_unchecked(i) })
}

/// More flexible variant of [`poly_array`]
#[inline(always)]
pub fn poly_array_t<F0, F1, T, const N: usize>(x: F0, coeffs: &[T; N]) -> F0
where
    F0: PolyInOut<F1> + From<F1>,
    F1: PolyNum,
    T: Clone + Into<F1>,
{
    poly_f_n::<F0, F1, _, N>(x, |i| unsafe { coeffs.get_unchecked(i).clone().into() })
}

/// Evaluate a polynomial for a slice of coefficients. May not be monomorphized.
///
/// To not be monomorphized means this function's codegen may be used for any number of coefficients,
/// and therefore contains branches. It will be faster to use [`poly_array`] instead if possible.
pub fn poly<F0: PolyInOut<F> + From<F>, F: PolyNum>(x: F0, coeffs: &[F]) -> F0 {
    poly_f_internal::<F0, F, _, 0>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

/// Evaluate a polynomial using a function to provide coefficients.
///
/// This function is more flexible than [`poly`] as it allows for the coefficients to be
/// generated on-the-fly. This can be useful for generating coefficients that are not
/// known at compile-time. However, this function may be slower than [`poly`] due to the
/// lack of monomorphization optimizations.
pub fn poly_f<F0: PolyInOut<F> + From<F>, F: PolyNum, G>(x: F0, n: usize, g: G) -> F0
where
    G: FnMut(usize) -> F,
{
    poly_f_internal::<F0, F, _, 0>(x, n, g)
}

/// Variation of [`poly_f`] that is monomorphized for a specific number of coefficients.
pub fn poly_f_n<F0: PolyInOut<F> + From<F>, F: PolyNum, G, const N: usize>(x: F0, g: G) -> F0
where
    G: FnMut(usize) -> F,
{
    poly_f_internal::<F0, F, _, 0>(x, N, g)
}

#[inline(always)]
#[rustfmt::skip]
fn poly_f_internal<F0: PolyInOut<F> + From<F>, F: PolyNum, G, const LENGTH: usize>(x: F0, n: usize, mut g: G) -> F0
where
    G: FnMut(usize) -> F,
{
    #![allow(clippy::wildcard_imports)]
    use polynomials::*;

    if LENGTH > 0 {
        unsafe { assume(n == LENGTH) };
    }

    macro_rules! poly {
        ($name:ident($($pows:expr),*; { $j:expr } + $c:ident[$($coeff:expr),*])) => {{
            $name($($pows,)* $($c($j + $coeff)),*)
        }};
    }

    // fast path for small input
    match n {
        0 => return F0::zero(),
        1 => return g(0).into(),
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

    let mut sum = F0::zero();

    // Use a hybrid Estrin/Horner algorithm
    let mut j = n;
    while j >= 16 {
        j -= 16;
        sum = fma::<F0,F0>(sum, x16, poly!(poly_15(x, x2, x4, x8; { j } + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])));
    }

    // handle remaining powers
    let (rmx, res) = match j {
        0  => return sum,
        1  => (x,              g(0).into()),
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

    fma::<F0,F0>(sum, rmx, res)
}

/// Evaluate a rational polynomial for an array of coefficients. Can be monomorphized.
///
/// To be monomorphized means a dedicated instance of this code will be generated for
/// the array of this length, removing many/all branches within the internal code that
/// other methods such as [`rational`] may require to support many lengths. This function will
/// be faster, put simply.
#[inline(always)]
pub fn rational_array<F0, F1, const P: usize, const Q: usize>(
    x: F0,
    numerator: &[F1; P],
    denomiator: &[F1; Q],
) -> F0
where
    F0: PolyRationalInOut<F1>,
    F1: PolyNum,
{
    rational_f_n::<F0, F1, _, _, P, Q>(
        x,
        |i| unsafe { *numerator.get_unchecked(i) },
        |i| unsafe { *denomiator.get_unchecked(i) },
    )
}

/// More flexible variant of [`rational_array`]
#[inline(always)]
pub fn rational_array_t<F0, F1, T, const P: usize, const Q: usize>(
    x: F0,
    numerator: &[T; P],
    denomiator: &[T; Q],
) -> F0
where
    F0: PolyRationalInOut<F1>,
    F1: PolyNum,
    T: Clone + Into<F1>,
{
    rational_f_n::<F0, F1, _, _, P, Q>(
        x,
        |i| unsafe { numerator.get_unchecked(i).clone().into() },
        |i| unsafe { denomiator.get_unchecked(i).clone().into() },
    )
}

/// Evaluate a rational polynomial for an array of coefficients. May not be monomorphized.
///
/// To not be monomorphized means this function's codegen may be used for any number of coefficients,
/// and therefore contains branches. It will be faster to use [`rational_array`] instead if possible.
pub fn rational<F0: PolyRationalInOut<F1>, F1: PolyNum>(
    x: F0,
    numerator: &[F1],
    denominator: &[F1],
) -> F0 {
    rational_f_internal::<F0, F1, _, _, 0, 0>(
        x,
        numerator.len(),
        denominator.len(),
        |i| unsafe { *numerator.get_unchecked(i) },
        |i| unsafe { *denominator.get_unchecked(i) },
    )
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
pub fn rational_f<F0, F1, N, D>(x: F0, p: usize, q: usize, numerator: N, denomiator: D) -> F0
where
    F0: PolyRationalInOut<F1>,
    F1: PolyNum,
    N: FnMut(usize) -> F1,
    D: FnMut(usize) -> F1,
{
    rational_f_internal::<F0, F1, _, _, 0, 0>(x, p, q, numerator, denomiator)
}

/// Variation of [`rational_f`] that is monomorphized for a specific number of coefficients.
#[inline]
pub fn rational_f_n<F0, F1, N, D, const P: usize, const Q: usize>(
    x: F0,
    numerator: N,
    denomiator: D,
) -> F0
where
    F0: PolyRationalInOut<F1>,
    F1: PolyNum,
    N: FnMut(usize) -> F1,
    D: FnMut(usize) -> F1,
{
    rational_f_internal::<F0, F1, _, _, P, Q>(x, P, Q, numerator, denomiator)
}

#[rustfmt::skip]
#[inline(always)]
#[allow(clippy::many_single_char_names)]
fn rational_f_internal<F0, F1, N, D, const P: usize, const Q: usize>(
    x: F0,
    p: usize,
    q: usize,
    mut numerator: N,
    mut denominator: D,
) -> F0
where
    F0: PolyRationalInOut<F1>,
    F1: PolyNum,
    N: FnMut(usize) -> F1,
    D: FnMut(usize) -> F1,
{
    let one = F0::one();

    // static or dynamic degree checks
    let high_degree = (P > 2 || Q > 2) || (P == 0 && Q == 0 && (p > 2 || q > 2));

    // if the length is greater than 2 (degree >= 2) the multiplication will be performed
    // anyway, and LLVM will reuse this result for the non-inverted polynomial below.
    if high_degree && (x * x) > one {
        if P > 0 { unsafe { assume(p == P) } }
        if Q > 0 { unsafe { assume(q == Q) } }

        // To prevent large values of x from exploding to infinity, we can replace x with z=1/x
        // and evaluate the polynomial in z to keep the powers of x within -1 and 1 where
        // floats are most accurate.

        let z = one / x;

        let n = poly_f_internal::<_,_, _, P>(z, p, |i| numerator(p - i - 1));
        let d = poly_f_internal::<_,_, _, Q>(z, q, |i| denominator(q - i - 1));

        let mut res = n / d;

        // no correction needed for same-degree rational polynomials
        if P == Q && (P > 0 || likely(p == q)) {
            return res;
        }

        // when the degree of the numerator and denominator are different, we need to correct
        // the result by shifting over the difference in degrees
        let (mut u, mut e) = if p < q { (z, q - p) } else { (x, p - q) };

        // `res = res * powi(u, e)` assuming e > 0
        // because e > 0 we can jump straight into the loop without a pre-check,
        // and rearrange some checks into a happy path.

        if P > 0 && Q > 0 {
            // this version optimizes better for static lengths
            loop {
                if e & 1 != 0 {
                    res = res * u;
                }

                e >>= 1;

                if e == 0 {
                    return res;
                }

                u = u * u;
            }
        } else {
            // and this version optimizes better for dynamic lengths
            loop {
                if e & 1 != 0 {
                    res = res * u;

                    if e == 1 {
                        return res;
                    }
                }

                e >>= 1;
                u = u * u;
            }
        }
    } else {
        poly_f_internal::<F0,F1, _, P>(x, p, numerator) / poly_f_internal::<F0,F1, _, Q>(x, q, denominator)
    }
}

#[inline(always)]
#[cold]
fn cold() {}

#[inline(always)]
#[rustfmt::skip]
fn likely(b: bool) -> bool {
    if !b { cold() } b
}

#[inline(always)]
unsafe fn assume(cond: bool) {
    if !cond {
        core::hint::unreachable_unchecked();
    }
}
