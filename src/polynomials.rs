//! Optimized fixed-degree polynomials for manual use.
//!
//! All of these polynomials use Estrin's scheme to reduce the dependency chain length
//! and encourage instruction-level parallelism, which has the potential to improve
//! performance despite the powers of X being required upfront.
//!
//! Powers of x are required, rather than computed internally, so they could be reused
//! between multiple polynomials.
//!
//! Unless you are micro-optimizing, it's recommended to use [`poly`](crate::poly),
//! [`poly_array`](crate::poly_array) or [`poly_f`](crate::poly_f).
//!
//! `poly_array` especially should optimize down to one of these as necessary.

#![allow(
    clippy::too_many_arguments,
    clippy::inline_always,
    clippy::similar_names
)]

use crate::{fma, PolyInOut, PolyNum};

#[inline(always)]
pub fn poly_1<F0: PolyInOut<F> + From<F>, F: PolyNum>(x: F0, c0: F, c1: F) -> F0 {
    let c0_up = c0.into();
    fma::<F0, F>(x, c1, c0_up)
}

#[inline(always)]
pub fn poly_2<F0: PolyInOut<F> + From<F>, F: PolyNum>(x: F0, x2: F0, c0: F, c1: F, c2: F) -> F0 {
    let c0_up = c0.into();
    fma(x2, c2, fma(x, c1, c0_up))
}

#[inline(always)]
pub fn poly_3<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0,
    x2: F0,
    c0: F,
    c1: F,
    c2: F,
    c3: F,
) -> F0 {
    // x^2 * (x * c3 + c2) + (x*c1 + c0)
    let c0_up = c0.into();
    let c2_up = c2.into();
    fma::<F0, F0>(x2, fma::<F0, F>(x, c3, c2_up), fma(x, c1, c0_up))
}

#[inline(always)]
pub fn poly_4<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0,
    x2: F0,
    x4: F0,
    c0: F,
    c1: F,
    c2: F,
    c3: F,
    c4: F,
) -> F0 {
    // x^4 * c4 + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    let c0_up = c0.into();
    let c2_up = c2.into();
    fma::<F0, F>(
        x4,
        c4,
        fma::<F0, F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    )
}

#[inline(always)]
pub fn poly_5<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0,
    x2: F0,
    x4: F0,
    c0: F,
    c1: F,
    c2: F,
    c3: F,
    c4: F,
    c5: F,
) -> F0 {
    // x^4 * (x * c5 + c4) + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    fma::<F0, F0>(
        x4,
        fma(x, c5, c4_up),
        fma::<F0, F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_6<F0: PolyInOut<F> + From<F>, F: PolyNum>(x: F0, x2: F0, x4: F0, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F) -> F0 {
    // x^4 * (x^2 * c6 + (x * c5 + c4)) + (x^2 * (x * c3 + c2) + (x * c1 + c0))
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    fma::<F0,F0>(x4,
        fma(x2, c6, fma(x, c5, c4_up)),
        fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_7<F0: PolyInOut<F> + From<F>, F: PolyNum>(x: F0, x2: F0, x4: F0, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    fma::<F0,F0>(x4,
        fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
        fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_8<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    fma(x8, c8, fma::<F0,F0>(x4,
        fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
        fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_9<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    fma::<F0,F0>(x8, fma(x, c9, c8_up), fma::<F0,F0>(x4,
        fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
        fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_10<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F,
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    fma::<F0,F0>(x8, fma(x2, c10, fma(x, c9, c8_up)), fma::<F0,F0>(x4,
        fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
        fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_11<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    let c10_up = c10.into();
    fma::<F0,F0>(x8,
        fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c9, c8_up)),
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
            fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_12<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F,
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    let c10_up = c10.into();
    fma::<F0,F0>(x8,
        fma(x4,
            c12,
            fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c9, c8_up)),
        ),
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
            fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_13<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F,
) -> F0 {
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    let c10_up = c10.into();
    let c12_up = c12.into();
    fma::<F0,F0>(x8,
        fma::<F0,F0>(x4,
            fma(x, c13, c12_up),
            fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c9, c8_up)),
        ),
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
            fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_14<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F, c14: F
) -> F0 {
    // (((C0+C1x) + (C2+C3x)x2) + ((C4+C5x) + (C6+C7x)x2)x4) + (((C8+C9x) + (C10+C11x)x2) + ((C12+C13x) + C14*x2)x4)x8
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    let c10_up = c10.into();
    let c12_up = c12.into();
    fma::<F0,F0>(x8,
        fma::<F0,F0>(x4,
            fma(x2, c14, fma(x, c13, c12_up)),
            fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c9, c8_up)),
        ),
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
            fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_15<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F, c14: F, c15: F
) -> F0 {
    // (((C0+C1x) + (C2+C3x)x2) + ((C4+C5x) + (C6+C7x)x2)x4) + (((C8+C9x) + (C10+C11x)x2) + ((C12+C13x) + (C14+C15x)x2)x4)x8
    let c0_up = c0.into();
    let c2_up = c2.into();
    let c4_up = c4.into();
    let c6_up = c6.into();
    let c8_up = c8.into();
    let c10_up = c10.into();
    let c12_up = c12.into();
    let c14_up = c14.into();
    fma::<F0,F0>(x8,
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c15, c14_up), fma(x, c13, c12_up)),
            fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c9, c8_up)),
        ),
        fma::<F0,F0>(x4,
            fma::<F0,F0>(x2, fma(x, c7, c6_up), fma(x, c5, c4_up)),
            fma::<F0,F0>(x2, fma(x, c3, c2_up), fma(x, c1, c0_up)),
        ),
    )
}

#[allow(dead_code)]
#[rustfmt::skip]
#[inline(always)]
pub fn poly_30<F0: PolyInOut<F> + From<F>, F: PolyNum>(
    x: F0, x2: F0, x4: F0, x8: F0, x16: F0,
    c00: F, c01: F, c02: F, c03: F, c04: F, c05: F, c06: F, c07: F, c08: F, c09: F, c10: F, c11: F, c12: F, c13: F, c14: F, c15: F,
    c16: F, c17: F, c18: F, c19: F, c20: F, c21: F, c22: F, c23: F, c24: F, c25: F, c26: F, c27: F, c28: F, c29: F, c30: F, c31: F
) -> F0 {
    let c00_up = c00.into();
    let c02_up = c02.into();
    let c04_up = c04.into();
    let c06_up = c06.into();
    let c08_up = c08.into();
    let c10_up = c10.into();
    let c12_up = c12.into();
    let c14_up = c14.into();
    let c16_up = c16.into();
    let c18_up = c18.into();
    let c20_up = c20.into();
    let c22_up = c22.into();
    let c24_up = c24.into();
    let c26_up = c26.into();
    let c28_up = c28.into();
    let c30_up = c30.into();
    fma::<F0,F0>(x16,
        fma::<F0,F0>(x8,
            fma::<F0,F0>(x4,
                fma::<F0,F0>(x2, fma(x, c31, c30_up), fma(x, c29, c28_up)),
                fma::<F0,F0>(x2, fma(x, c27, c26_up), fma(x, c25, c24_up)),
            ),
            fma::<F0,F0>(x4,
                fma::<F0,F0>(x2, fma(x, c23, c22_up), fma(x, c21, c20_up)),
                fma::<F0,F0>(x2, fma(x, c19, c18_up), fma(x, c17, c16_up)),
            ),
        ),
        fma::<F0,F0>(x8,
            fma::<F0,F0>(x4,
                fma::<F0,F0>(x2, fma(x, c15, c14_up), fma(x, c13, c12_up)),
                fma::<F0,F0>(x2, fma(x, c11, c10_up), fma(x, c09, c08_up)),
            ),
            fma::<F0,F0>(x4,
                fma::<F0,F0>(x2, fma(x, c07, c06_up), fma(x, c05, c04_up)),
                fma::<F0,F0>(x2, fma(x, c03, c02_up), fma(x, c01, c00_up)),
            ),
        )
    )
}
