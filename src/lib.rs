use num_traits::Float;

#[inline(always)]
fn fma<F: Float>(x: F, m: F, a: F) -> F {
    #[cfg(feature = "fma")]
    return x.mul_add(m, a);

    #[cfg(not(feature = "fma"))]
    return x * m + a;
}

pub mod poly;

#[inline]
pub fn horners_method<F: Float>(x: F, coeffs: &[F]) -> F {
    horners_method_f(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

#[inline]
pub fn poly_array<F: Float, const N: usize>(x: F, coeffs: &[F; N]) -> F {
    poly_f::<F, _, N>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

#[inline]
pub fn poly<F: Float>(x: F, coeffs: &[F]) -> F {
    poly_f::<F, _, 0>(x, coeffs.len(), |i| unsafe { *coeffs.get_unchecked(i) })
}

#[inline(always)]
pub fn horners_method_f<F: Float, G>(x: F, mut n: usize, mut g: G) -> F
where
    G: FnMut(usize) -> F,
{
    let mut sum = F::zero();

    while n > 0 {
        n -= 1;
        sum = fma(sum, x, g(n));
    }

    sum
}

#[inline(always)]
#[rustfmt::skip]
pub fn poly_f<F: Float, G, const MONO_HACK: usize>(x: F, n: usize, mut g: G) -> F
where
    G: FnMut(usize) -> F,
{
    use poly::*;

    const MAX_DEGREE_P0: usize = 16;

    // fast path for small input
    if n <= MAX_DEGREE_P0 {
        return if n < 5 {
            match n {
                0 => F::zero(),
                1 => g(0),
                2 => poly_1(x, g(0), g(1)),
                3 => poly_2(x, x * x, g(0), g(1), g(2)),
                4 => poly_3(x, x * x, g(0), g(1), g(2), g(3)),
                _ => unsafe { core::hint::unreachable_unchecked() }
            }
        } else {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            let (g0, g1, g2, g3, g4) = (g(0), g(1), g(2), g(3), g(4));

            match n {
                5 =>  poly_4 (x, x2, x4,     g0, g1, g2, g3, g4),
                6 =>  poly_5 (x, x2, x4,     g0, g1, g2, g3, g4, g(5)),
                7 =>  poly_6 (x, x2, x4,     g0, g1, g2, g3, g4, g(5), g(6)),
                8 =>  poly_7 (x, x2, x4,     g0, g1, g2, g3, g4, g(5), g(6), g(7)),
                9 =>  poly_8 (x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8)),
                10 => poly_9 (x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9)),
                11 => poly_10(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10)),
                12 => poly_11(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10), g(11)),
                13 => poly_12(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12)),
                14 => poly_13(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13)),
                15 => poly_14(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13), g(14)),
                16 => poly_15(x, x2, x4, x8, g0, g1, g2, g3, g4, g(5), g(6), g(7), g(8), g(9), g(10), g(11), g(12), g(13), g(14), g(15)),
                _ => unsafe { core::hint::unreachable_unchecked() }
            }
        };
    }

    let xmd = x.powi(MAX_DEGREE_P0 as i32);

    let mut sum = F::zero();

    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polys() {
        #[rustfmt::skip]
        let c = [
            0.9066094402137101, 0.7030666449646632, 0.8062843184510005, 1.4354479997076703, 1.1700851966666594,
            1.0036799628327977, 0.669178962803656, 0.7728758968389648, 0.5606587385173203, 1.0939290310925256,
            0.8620002023538906, 1.2530914565673503, 1.4918792702029815, 1.3154976283644524, 1.3564397050359411,
            1.0271024168686784, 1.405690756664578, 0.5449121493513336, 0.9862179238638533, 0.9124457978499287,
            0.8732207167879933, 0.6630588917237896, 0.5904674982257736, 1.4169918094580403, 0.958839837872578,
            0.5505474299309041, 0.8383676032996494, 0.9596512540091879, 0.6559726022409615, 1.0826517080111482,
            1.3846791166569572, 1.3762199390279588, 0.6807699410480192, 0.9768566731838964, 1.2572212915635828,
            0.701803747744993, 0.8273020543751974, 1.4638922915963615, 1.348778424905363, 1.3457337576150634,
            1.1274404084913705, 0.6266756469558616,
        ];

        assert!((horners_method(0.2, &c) - 1.09320587687) < 1e-10);
        assert!((poly(0.2, &c) - 1.09320587687) < 1e-10);

        assert!((horners_method(-0.4, &c[..4]) - 0.662519601199) < 1e-10);
        assert!((poly(-0.4, &c[..4]) - 0.662519601199) < 1e-10);
    }
}
