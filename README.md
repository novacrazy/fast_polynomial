fast_polynomial
===============

This crate implements a hybrid Estrin's/Horner's method suitable for evaluating polynomials _fast_
by exploiting instruction-level parallelism.

## **Important Note About Fused Multiply-Add**

FMA is only used by Rust if your binary is compiled with the appropriate Rust flags:

`RUSTFLAGS="-C target-feature=+fma"`

or
```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-feature=+fma"]
```

otherwise separate multiply and addition operations are used.

## Motivation

Consider the following simple polynomial evaluation function:

```rust
fn horners_method(x: f32, coefficients: &[f32]) -> f32 {
    let mut sum = 0.0;
    for coeff in coefficients.iter().rev().copied() {
        sum = x * sum + coeff;
    }
    sum
}

assert_eq!(horners_method(0.5, &[1.0, 0.3, 0.4, 1.6]), 1.45);
```

Simple and clean, this is [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method). However,
note that each iteration relies on the result of the previous, creating a dependency chain that cannot
be parallelized, and must be executed sequentially:

```asm
vxorps      %xmm1,    %xmm1, %xmm1
vfmadd213ss 12(%rdx), %xmm0, %xmm1 /* Note the reuse of xmm1 for all vfmadd213ss */
vfmadd213ss 8(%rdx),  %xmm0, %xmm1
vfmadd213ss 4(%rdx),  %xmm0, %xmm1
vfmadd213ss (%rdx),   %xmm1, %xmm0
```

[Estrin's Scheme](https://en.wikipedia.org/wiki/Estrin's_scheme) is a way of organizing polynomial calculations
such that they can compute parts of the polynomial in parallel using instruction-level parallelism. ILP is where
a modern CPU can queue up multiple calculations at once so long as they don't rely on each other.

For example, `(a + b) + (c + d)` will likely compute each parenthesized half of this
expression using separate registers, at the same time.

This crate leverages this for all polynomials up to degree-15, at which point it switches over to a hybrid method
that can process arbitrarily high degree polynomials up to 15 coefficients at a time.

With the above example with 4 coefficients, using `poly_array` will generate this assembly:
```asm
vmovss      4(%rdx),  %xmm1
vmovss      12(%rdx), %xmm3
vmulss      %xmm0,    %xmm0, %xmm2
vfmadd213ss 8(%rdx),  %xmm0, %xmm3
vfmadd213ss (%rdx),   %xmm0, %xmm1
vfmadd231ss %xmm3,    %xmm2, %xmm1
```

Note that it uses multiple xmm registers, as the first two `vfmadd213ss` instructions will run in parallel. The `vmulss` instruction
will also likely run in parallel to those FMAs. Despite being more individual instructions, because they run in parallel on hardware,
this will be significantly faster.

## Rational Polynomials

`fast_polynomial` supports evaluating rational polynomials such as those found in [Padé approximations](https://en.wikipedia.org/wiki/Pad%C3%A9_approximant), but with an **important note**: To avoid powers of the input `x` exploding, we perform a technique where we replace `x`
with `z = 1/x` and evaluate the polynomial effectively in reverse:

<details>
<summary>Click to open rendered example</summary>

If this isn't rendered for you, [view it on the GitHub readme](https://github.com/novacrazy/fast_polynomial/blob/main/README.md#rational-polynomials).

```math
\begin{align}
    \frac{a_0 + a_1 x + a_2 x^2}{b_0 + b_1 x + b_2 x^2} &= \frac{a_0 + a_1 z^{-1} + a_2 z^{-2}}{b_0 + b_1 z^{-1} + b_2 z^{-2}} \\
        &= \frac{a_0 z^2 + a_1 z + a_2}{b_0 z^2 + b_1 z + b_2} \\
        &= \frac{a_2 + a_1 z + a_0 z^2}{b_2 + b_1 z + b_0 z^2} \\
\end{align}
```

</details>

However, should the numerator and denominator have different degrees, an additional correction step is required to shift over the degrees to match, which can reduce performance and potentially accuracy, so it should be avoided. It may genuinely be faster to pad your polynomials to the same degree, especially if using `rational_array` to avoid excessive codegen.

## Other Disadvantages

Estrin's scheme is slightly more numerically unstable for very high-degree polynomials. However, using FMA and the
provided rational polynomial evaluation routines both improve numerical stability where possible.

## Additional notes

Using `poly_array` can be significantly more performant for fixed-degree polynomials. In optimized builds,
the monomorphized codegen will be nearly ideal and avoid unnecessary branching.

However, should you need to evaluate multiple polynomials with the same X value, the `polynomials` module
exists to provide direct fixed-degree functions that allow the reuse of powers of X up to degree-15.

## Cargo Features

The `std` (default) and `libm` crate features are passed through to `num-traits`.