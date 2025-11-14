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

#![allow(clippy::too_many_arguments, missing_docs)]

use crate::PolyNum;

#[inline(always)]
pub fn poly_1<F: PolyNum>(x: F, c0: F, c1: F) -> F {
    x.mul_add(c1, c0)
}

#[inline(always)]
pub fn poly_2<F: PolyNum>(x: F, x2: F, c0: F, c1: F, c2: F) -> F {
    x2.mul_add(c2, x.mul_add(c1, c0))
}

#[inline(always)]
pub fn poly_3<F: PolyNum>(x: F, x2: F, c0: F, c1: F, c2: F, c3: F) -> F {
    // x^2 * (x * c3 + c2) + (x*c1 + c0)
    x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0))
}

#[inline(always)]
pub fn poly_4<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F) -> F {
    // x^4 * c4 + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    x4.mul_add(c4, x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)))
}

#[inline(always)]
pub fn poly_5<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F) -> F {
    // x^4 * (x * c5 + c4) + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    x4.mul_add(
        x.mul_add(c5, c4),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_6<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F) -> F {
    // x^4 * (x^2 * c6 + (x * c5 + c4)) + (x^2 * (x * c3 + c2) + (x * c1 + c0))
    x4.mul_add(
        x2.mul_add(c6, x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_7<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F) -> F {
    x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_8<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F
) -> F {
    x8.mul_add(c8, x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_9<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F
) -> F {
    x8.mul_add(x.mul_add(c9, c8), x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_10<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F,
) -> F {
    x8.mul_add(x2.mul_add(c10, x.mul_add(c9, c8)), x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_11<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F
) -> F {
    x8.mul_add(
        x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_12<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F,
) -> F {
    x8.mul_add(
        x4.mul_add(
            c12,
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_13<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F,
) -> F {
    x8.mul_add(
        x4.mul_add(
            x.mul_add(c13, c12),
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_14<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F, c14: F
) -> F {
    // (((C0+C1x) + (C2+C3x)x2) + ((C4+C5x) + (C6+C7x)x2)x4) + (((C8+C9x) + (C10+C11x)x2) + ((C12+C13x) + C14*x2)x4)x8
    x8.mul_add(
        x4.mul_add(
            x2.mul_add(c14, x.mul_add(c13, c12)),
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_15<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F, c14: F, c15: F
) -> F {
    // (((C0+C1x) + (C2+C3x)x2) + ((C4+C5x) + (C6+C7x)x2)x4) + (((C8+C9x) + (C10+C11x)x2) + ((C12+C13x) + (C14+C15x)x2)x4)x8
    x8.mul_add(
        x4.mul_add(
            x2.mul_add(x.mul_add(c15, c14), x.mul_add(c13, c12)),
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_31<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F, x16: F,
    c00: F, c01: F, c02: F, c03: F, c04: F, c05: F, c06: F, c07: F, c08: F, c09: F, c10: F, c11: F, c12: F, c13: F, c14: F, c15: F,
    c16: F, c17: F, c18: F, c19: F, c20: F, c21: F, c22: F, c23: F, c24: F, c25: F, c26: F, c27: F, c28: F, c29: F, c30: F, c31: F
) -> F {
    x16.mul_add(
        x8.mul_add(
            x4.mul_add(
                x2.mul_add(x.mul_add(c31, c30), x.mul_add(c29, c28)),
                x2.mul_add(x.mul_add(c27, c26), x.mul_add(c25, c24)),
            ),
            x4.mul_add(
                x2.mul_add(x.mul_add(c23, c22), x.mul_add(c21, c20)),
                x2.mul_add(x.mul_add(c19, c18), x.mul_add(c17, c16)),
            ),
        ),
        x8.mul_add(
            x4.mul_add(
                x2.mul_add(x.mul_add(c15, c14), x.mul_add(c13, c12)),
                x2.mul_add(x.mul_add(c11, c10), x.mul_add(c09, c08)),
            ),
            x4.mul_add(
                x2.mul_add(x.mul_add(c07, c06), x.mul_add(c05, c04)),
                x2.mul_add(x.mul_add(c03, c02), x.mul_add(c01, c00)),
            ),
        )
    )
}
