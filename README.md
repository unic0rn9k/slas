<div align="center">

## SLAS
*Static Linear Algebra System*

[![Crates.io](https://img.shields.io/crates/v/slas?logo=rust&style=flat-square)](https://crates.io/crates/slas)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/slas/Tests?label=tests&logo=github&style=flat-square)](https://github.com/unic0rn9k/slas/actions/workflows/rust.yml)
[![Docs](https://img.shields.io/docsrs/slas/latest?logo=rust&style=flat-square)](https://docs.rs/slas/latest/slas/)
[![Donate on paypal](https://img.shields.io/badge/paypal-donate-1?style=flat-square&logo=paypal&color=blue)](https://www.paypal.com/paypalme/unic0rn9k/5usd)

</div>

Provides statically allocated vector, matrix and tensor types, for interfacing with blas/blis, in a performant manor, using copy-on-write (aka cow) behavior.

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
The idea is simply that allocations (and time) can be saved, by figuring out when to copy at runtime instead of at compiletime.
This can be memory inefficient at times (as an enum takes the size of its largest field + tag size), but I'm planing on making ways around this in the future.

**NOTICE:** If you're using the git version of slas, you can now use `StaticVecRef`'s instead of `StaticCowVecs`, when you don't want the cow behavior.

 ### In code...
```rust
use slas::prelude::*;

let source: Vec<f32> = vec![1., 2., 3.];
let mut v = moo![_ source.as_slice()];

// Here we mutate v,
// so the content of source will be copied into v before the mutation occours.
v[0] = 0.;

assert_eq!(**v, [0., 2., 3.]);
assert_eq!(source, vec![1., 2., 3.]);
```

The borrow checker won't allow mutating `source` after `v` is created, because assignment to borrowed values is not allowed.
This can be a problem in some situations.

```rust
use slas::prelude::*;

let mut source: Vec<f32> = vec![1., 2., 3.];
let mut v = unsafe { StaticCowVec::<f32, 3>::from_ptr(source.as_ptr()) };

// Here we can mutate source, because v was created from a raw pointer.
source[1] = 3.;
v[0] = 0.;
source[2] = 4.;

assert_eq!(**v, [0., 3., 3.]);
assert_eq!(source, vec![1., 3., 4.]);
```
In the example above, you can see `v` changed value the first time `source` was mutated, but not the second time.
This is because `v` was copied when it was mutated at the line after the first mutation of `source`.

### Matricies, tensors and other mathematical types
At the moment the way I want to implement these types, causes a compiler crash, when trying to create 2 objects with the same shape.
For now I'm going to try to create a temporary, and more stable, way of dealing with these variations of static multi dimensional arrays.

As of now there is a Matrix type, but no tensor type on the master branch.
The stable matricies are very basic, as I hopefully will be able to replace them with a more generic tensor type soon...

```rust
use slas::{prelude::*, matrix::Matrix};

let m: Matrix<f32, 2, 3> = [
 1., 2., 3.,
 4., 5., 6.
].matrix_ref();

assert!(m[[1, 0]] == 2.);

let k: Matrix<f32, 3, 2> = moo![f32: 0..6].into();

println!("Product of {:?} and {:?} is {:?}", m, k, m * k);
```

If you want a look at whats to come in the future,
you can go [here](https://github.com/unic0rn9k/slas/tree/experimental/src/experimental)
for some *very* experimental source code for the project.

### Why not just use ndarray (or alike)?
Slas can be faster than ndarray in some specific use cases, like when having to do a lot of allocations, or when using referenced data in vector operations.
Besides slas should always be atleast as fast as ndarray, so it can't hurt.

Statical allocation and the way slas cow behavior works with the borrow checker,
also means that you might catch a lot of bugs at compiletime,
where ndarray most of the time will let you get away with pretty much anything.

### Installation
Slas depends on blas, and currently only supports using blis.
In the future you will have to choose your own blas provider, and instructions for doing so will be added here.

On the crates.io version of slas (v0.1.0 and 0.1.1) blis is compiled automatically.

For now, if you want to use the git version of slas, you need to install blis on your system.
- On Arch linux `blis-cblas` v0.7.0 from the aur has been tested and works fine.
- On Debian you can simply run `apt install libblis-dev`.

### General info...
- Slas is still in very early days, and is subject to a lot of breaking changes.
- The `StaticCowVec` type implements `deref` and `deref_mut`, so any method implemented for `[T;LEN]` is also implemented for `StaticCowVec`.
- [Benchmarks, tests and related](https://github.com/unic0rn9k/slas/tree/master/tests)

### TODO
- ~~Rust version of blas functions allowing for loop unrolling - also compile time checks for choosing fastest function~~
- Make less terrible benchmarks
- Feature support for conversion between [ndarray](lib.rs/ndarray) types
- Allow for use on stable channel - perhabs with a stable feature
- Implement stable tensors - perhabs for predefined dimensions with a macro
- ~~Make StaticCowVec backed by a union - so that vectors that are always owned can also be supported (useful for memory critical systems, fx. embeded devices).~~
- ~~Modular backends - [like in coaster](https://github.com/spearow/juice/tree/master/coaster)~~
    - GPU support - maybe with cublas
    - ~~pure rust support - usefull for irust and jupyter support.~~
- Write unit tests to make sure unsafe functions can't produce ub.

License: Apache-2.0
