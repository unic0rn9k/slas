# How to replicate
You can test and benchmark slas on your local machine by running
`cargo test -p tests` and `cargo bench -p tests`
respectively, in the repository root.

If you want to compare performance of slas and [ndarray](https://lib.rs/ndarray) you can run 

`cargo bench -j8 -p tests --features tests/vsndarray`


# Results and comparing with ndarray
**This is a bad benchmark, and I know it. This is mostly just here as a place holder until I a) have more stuff to benchmark, and b) I actually start optimizing the code beyond the initial performance focused design.**

On my machine slas got slightly better performance when benchmarking dot products of vectors larger than 300 elements.
The performance advantage was only present when allocating inside the benchmark loop.
This is reasonable, since slas and ndarray both call blis. So the only part of slas that is faster is allocation, which is to be expected, since it is entirely statically allocated, whereas ndarray is dynamically allocated.

``` text
test vs_ndarray::ndarray::dot ... bench:       1,359 ns/iter (+/- 217)
test vs_ndarray::slas::dot    ... bench:       1,227 ns/iter (+/- 175)
```
*Benchmark for 500 elements and 32-bit floats using blis as backend for both slas and ndarray.*


