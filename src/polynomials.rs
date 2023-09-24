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

#![allow(clippy::too_many_arguments)]

use crate::{fma, PolyNum};

#[inline(always)]
pub fn poly_1<F: PolyNum>(x: F, c0: F, c1: F) -> F {
    fma(x, c1, c0)
}

#[inline(always)]
pub fn poly_2<F: PolyNum>(x: F, x2: F, c0: F, c1: F, c2: F) -> F {
    fma(x2, c2, fma(x, c1, c0))
}

#[inline(always)]
pub fn poly_3<F: PolyNum>(x: F, x2: F, c0: F, c1: F, c2: F, c3: F) -> F {
    // x^2 * (x * c3 + c2) + (x*c1 + c0)
    fma(x2, fma(x, c3, c2), fma(x, c1, c0))
}

#[inline(always)]
pub fn poly_4<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F) -> F {
    // x^4 * c4 + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    fma(x4, c4, fma(x2, fma(x, c3, c2), fma(x, c1, c0)))
}

#[inline(always)]
pub fn poly_5<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F) -> F {
    // x^4 * (x * c5 + c4) + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    fma(x4, fma(x, c5, c4), fma(x2, fma(x, c3, c2), fma(x, c1, c0)))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_6<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F) -> F {
    // x^4 * (x^2 * c6 + (x * c5 + c4)) + (x^2 * (x * c3 + c2) + (x * c1 + c0))
    fma(x4,
        fma(x2, c6, fma(x, c5, c4)),
        fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_7<F: PolyNum>(x: F, x2: F, x4: F, c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F) -> F {
    fma(x4,
        fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
        fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_8<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F
) -> F {
    fma(x8, c8, fma(x4,
        fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
        fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_9<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F
) -> F {
    fma(x8, fma(x, c9, c8), fma(x4,
        fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
        fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_10<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F,
) -> F {
    fma(x8, fma(x2, c10, fma(x, c9, c8)), fma(x4,
        fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
        fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_11<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F
) -> F {
    fma(x8,
        fma(x2, fma(x, c11, c10), fma(x, c9, c8)),
        fma(x4,
            fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
            fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_12<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F,
) -> F {
    fma(x8,
        fma(x4,
            c12,
            fma(x2, fma(x, c11, c10), fma(x, c9, c8)),
        ),
        fma(x4,
            fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
            fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_13<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F,
    c0: F, c1: F, c2: F, c3: F, c4: F, c5: F, c6: F, c7: F, c8: F, c9: F, c10: F, c11: F, c12: F, c13: F,
) -> F {
    fma(x8,
        fma(x4,
            fma(x, c13, c12),
            fma(x2, fma(x, c11, c10), fma(x, c9, c8)),
        ),
        fma(x4,
            fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
            fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
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
    fma(x8,
        fma(x4,
            fma(x2, c14, fma(x, c13, c12)),
            fma(x2, fma(x, c11, c10), fma(x, c9, c8)),
        ),
        fma(x4,
            fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
            fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
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
    fma(x8,
        fma(x4,
            fma(x2, fma(x, c15, c14), fma(x, c13, c12)),
            fma(x2, fma(x, c11, c10), fma(x, c9, c8)),
        ),
        fma(x4,
            fma(x2, fma(x, c7, c6), fma(x, c5, c4)),
            fma(x2, fma(x, c3, c2), fma(x, c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_30<F: PolyNum>(
    x: F, x2: F, x4: F, x8: F, x16: F,
    c00: F, c01: F, c02: F, c03: F, c04: F, c05: F, c06: F, c07: F, c08: F, c09: F, c10: F, c11: F, c12: F, c13: F, c14: F, c15: F,
    c16: F, c17: F, c18: F, c19: F, c20: F, c21: F, c22: F, c23: F, c24: F, c25: F, c26: F, c27: F, c28: F, c29: F, c30: F, c31: F
) -> F {
    fma(x16,
        fma(x8,
            fma(x4,
                fma(x2, fma(x, c31, c30), fma(x, c29, c28)),
                fma(x2, fma(x, c27, c26), fma(x, c25, c24)),
            ),
            fma(x4,
                fma(x2, fma(x, c23, c22), fma(x, c21, c20)),
                fma(x2, fma(x, c19, c18), fma(x, c17, c16)),
            ),
        ),
        fma(x8,
            fma(x4,
                fma(x2, fma(x, c15, c14), fma(x, c13, c12)),
                fma(x2, fma(x, c11, c10), fma(x, c09, c08)),
            ),
            fma(x4,
                fma(x2, fma(x, c07, c06), fma(x, c05, c04)),
                fma(x2, fma(x, c03, c02), fma(x, c01, c00)),
            ),
        )
    )
}
