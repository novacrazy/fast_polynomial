fast_polynomial
===============

This crate implements a hybrid Estrin's/Horner's method suitable for evaluating high-degree polynomials _fast_
by exploiting instruction-level parallelism.

By default, the `fma` crate feature is enabled, which uses the `mul_add` function from `num-traits` to improve
both performance and accuracy. However, when a dedicated FMA (Fused Multiply-Add) instruction is not available,
Rust will substitute it with a function call that emulates it, decreasing performance. Set `default-features = false`
to fallback to split multiply and addition operations. If a reliable method of autodetecting this at compile time
is found, I will update the crate to do that instead, but `#[cfg(target_feature = "fma")]` does not work correctly.

The `std` (default) and `libm` crate features are passed through to `num-traits`.