<div align="center">

## SLAS

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/slas/Tests?label=tests&style=flat-square)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)
[![Donate on paypal](https://img.shields.io/badge/paypal-donate-1?style=flat-square&logo=paypal&color=blue)](https://www.paypal.com/paypalme/unic0rn9k/5usd)

</div>

Static Linear Algebra System.

Provides statically allocated vector, matrix and tensor structs, for interfacing with blas/blis, in a performant manor, using cows (Copy On Write).

### Example
```rust
use slas::prelude::*;
let a = moo![f32: 1, 2, 3.2];
let b = moo![f32: 3, 0.4, 5];
println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
```
[More example code here.](https://github.com/unic0rn9k/slas/blob/master/tests/src/main.rs)

### Test and Benchmark it yourself!
You can get benchmark results and tests by running
`cargo test -p tests` and `cargo bench -p tests`
in the root of the repository.

### Todo before publishing 🎉
- ~~Move ./experimental to other branch~~
- Implement stable tensors, perhabs for predefined dimensions with a macro
- ~~Implement Debug for matrix~~
- ~~Fix matrix api (Column and row specification is weird)~~
- Write documentation
- Benchmark against ndarray (and maybe others?)
- Allow for use on stable channel, perhabs with a stable feature
