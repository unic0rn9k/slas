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

### What is a COW?
The copy-on-write functionality is inspired by [std::borrow::cow](https://doc.rust-lang.org/std/borrow/enum.Cow.html).
The idea is simply that its easier to figure out when its most performant to copy vs referencing at runtime.

#### How it work in practise?
```rust
let source: Vec<f32> = vec![1., 2., 3.];
let mut v = moo![_ source.as_slice()];

// Here we mutate v,
// so the content of source will be copied into v before the mutation occours.
v[0] = 0.;

assert_eq!(*v, [0., 2., 3.]);
assert_eq!(source, vec![1., 2., 3.]);
```

The borrow checker won't allow mutating `source` after `v` is created, because assignment to borrowed values is not allowed.
This can be a problem in some situations.

```rust
let mut source: Vec<f32> = vec![1., 2., 3.];
let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };

// Here we can mutate source, because v was created from a raw pointer.
source[1] = 3.;
v[0] = 0.;
source[2] = 4.;

assert_eq!(*v, [0., 3., 3.]);
assert_eq!(source, vec![1., 3., 4.]);
```
In the example above, you can see `v` changed value the first time `source` was mutated, but not the second time.
This is because `v` was copied when it was mutated at the line after the first mutation of `source`.

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
