# slas

Static Linear Algebra System. A object oriented expansion of blas/blis, that allow for statically allocated cow (Copy On Write) vectors, matricies and tensors.

## Example
```rust
use slas::prelude::*;
let a = moo![f32: 1., 2., 3.];
let b = moo![f32: 3., 4., 5.];
println!("Dot product of {:?} and {:?} is {:?}", a, b, a.dot(&b));
```
