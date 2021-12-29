# slas

[![Workflow Status](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml/badge.svg)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)

Static Linear Algebra System.

Provides statically allocated vector, matrix and tensor structs for interfacing with blas/blis, in a performant manor, using cows (Copy On Write).

### Example
```rust
use slas::prelude::*;
let a = moo![f32: 1, 2, 3.2];
let b = moo![f32: 3, 0.4, 5];
println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
```
[More example code here.](https://github.com/unic0rn9k/slas/blob/master/tests/src/main.rs)

### Todo before publishing 🎉
- Move ./experimental to other branch
- Implement stable tensors, perhabs for predefined dimensions with a macro
- Implement Debug for matrix
- Fix matrix api (Column and row specification is weird)
