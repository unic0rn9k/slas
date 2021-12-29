# slas

Static Linear Algebra System. A object oriented expansion of blas/blis, that allow for statically allocated cow (Copy On Write) vectors, matricies and tensors.

### Status
[![Rust](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml/badge.svg)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)

### Example
```rust
use slas::prelude::*;
let a = moo![f32: 1, 2, 3.2];
let b = moo![f32: 3, 0.4, 5];
println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
```
