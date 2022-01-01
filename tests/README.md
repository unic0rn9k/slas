# How to replicate
You can test and benchmark slas on your local machine by running
`cargo test -p tests` and `cargo bench -p tests`
respectively, in the repository root.

If you want to compare performance of slas, [ndarray](https://lib.rs/ndarray) and [nalgebra](https://nalgebra.org) you can run 

`cargo bench -j8 -p tests --features tests/versus`


# Results and comparing with ndarray
**This is a bad benchmark, and I know it. This is mostly just here as a place holder until I a) have more stuff to benchmark, and b) I actually start optimizing the code beyond the initial performance focused design.**

On my machine slas got slightly better performance when benchmarking dot products of vectors larger than 300 elements.
The performance advantage was only present when allocating inside the benchmark loop.
This is reasonable, since slas and ndarray both call blis. So the only part of slas that is faster is allocation, which is to be expected, since it is entirely statically allocated, whereas ndarray is dynamically allocated.

``` text
test vs_ndarray::ndarray::dot   ... bench:       5,719 ns/iter (+/- 1,185)
test vs_ndarray::slas::dot      ... bench:       5,455 ns/iter (+/- 755)
```
*Benchmark for 500 elements and 32-bit floats using blis as backend for both slas and ndarray.*


