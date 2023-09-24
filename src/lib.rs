#![no_std]

use core::ops::{Add, Mul};
use num_traits::{MulAdd, Zero};

/// The minimum requires functionality required for a number to evaluated in a polynomial.
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

#[inline(always)]
fn fma<F: PolyNum>(x: F, m: F, a: F) -> F {
    #[cfg(feature = "fma")]
    return x.mul_add(m, a);

    #[cfg(not(feature = "fma"))]
    return x * m + a;
}

pub mod polynomials;

/// Evaluate a polynomial for an array of coefficients. Can be monomorphized.
///
/// To be monomorphized means a dedicated instance of this code will be generated for
/// the array of this length, removing many/all branches within the internal code that
/// invocations such as [`poly`] may require to support many lengths. This function will
/// be faster.
#[inline]
pub fn poly_array<F: PolyNum, const N: usize>(x: F, coeffs: &[F; N]) -> F {
    poly_f_internal::<F, _, N>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

/// Evaluate a polynomial for a slice of coefficients. May not be monomorphized.
///
/// To not be monomorphized means this function may be used for any number of coefficients,
/// and therefore contains branches. It may be faster to use [`poly_array`] instead.
pub fn poly<F: PolyNum>(x: F, coeffs: &[F]) -> F {
    poly_f_internal::<F, _, 0>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

/// Evaluate a polynomial using a function to provide coefficients.
#[inline(always)]
pub fn poly_f<F: PolyNum, G>(x: F, n: usize, g: G) -> F
where
    G: FnMut(usize) -> F,
{
    poly_f_internal::<F, _, 0>(x, n, g)
}

#[inline(always)]
#[rustfmt::skip]
fn poly_f_internal<F: PolyNum, G, const LENGTH: usize>(x: F, n: usize, mut g: G) -> F
where
    G: FnMut(usize) -> F,
{
    use polynomials::*;

    // if LENGTH is used, assume n = LENGTH to improve codegen
    if LENGTH != 0 && n != LENGTH {
        unsafe { core::hint::unreachable_unchecked() };
    }

    const MAX_DEGREE_P0: usize = 16;

    // fast path for small input
    match n {
        0 => return F::zero(),
        1 => return g(0),
        2 => return poly_1(x, g(0), g(1)),
        3 => return poly_2(x, x * x, g(0), g(1), g(2)),
        4 => return poly_3(x, x * x, g(0), g(1), g(2), g(3)),
        _ => {}
    }

    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    match n {
        5 =>  return poly_4 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4)),
        6 =>  return poly_5 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5)),
        7 =>  return poly_6 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5), g(6)),
        8 =>  return poly_7 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7)),
        9 =>  return poly_8 (x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8)),
        10 => return poly_9 (x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9)),
        11 => return poly_10(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10)),
        12 => return poly_11(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11)),
        13 => return poly_12(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12)),
        14 => return poly_13(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13)),
        15 => return poly_14(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13), g(14)),
        16 => return poly_15(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13), g(14), g(15)),
        _ => {}
    }

    let x16 = x8 * x8;

    let xmd = x16; // x.powi(MAX_DEGREE_P0 as i32);

    let mut sum = F::zero();

    // Use a hybrid Estrin/Horner algorithm
    let mut j = n;
    while j >= MAX_DEGREE_P0 {
        macro_rules! poly {
            ($name:ident($($pows:ident),*; $j:ident + $c:ident[$($coeff:expr),*])) => {{
                $name($($pows,)* $($c($j + $coeff)),*)
            }};
        }

        j -= MAX_DEGREE_P0;
        sum = fma(sum, xmd, poly!(poly_15(x, x2, x4, x8; j + g[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])));
    }

    // handle remaining powers
    let (rmx, res) = match j {
        0  => return sum,
        1  => (x,                                  g(0)),
        2  => (x2,          poly_1 (x,             g(0), g(1))),
        3  => (x2*x,        poly_2 (x, x2,         g(0), g(1), g(2))),
        4  => (x4,          poly_3 (x, x2,         g(0), g(1), g(2), g(3))),
        5  => (x4*x,        poly_4 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4))),
        6  => (x4*x2,       poly_5 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5))),
        7  => (x4*x2*x,     poly_6 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5), g(6))),
        8  => (x8,          poly_7 (x, x2, x4,     g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7))),
        9  => (x8*x,        poly_8 (x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8))),
        10 => (x8*x2,       poly_9 (x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9))),
        11 => (x8*x2*x,     poly_10(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10))),
        12 => (x8*x4,       poly_11(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11))),
        13 => (x8*x4*x,     poly_12(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12))),
        14 => (x8*x4*x2,    poly_13(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13))),
        15 => (x8*x4*x2*x,  poly_14(x, x2, x4, x8, g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13), g(14))),
        _  => unsafe { core::hint::unreachable_unchecked() }
    };

    fma(sum, rmx, res)
}
