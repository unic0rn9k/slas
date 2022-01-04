# How to replicate
You can test and benchmark slas on your local machine by running
`cargo test -p tests` and `cargo bench -p tests`
respectively, in the repository root.

If you want to compare performance of slas, [ndarray](https://lib.rs/ndarray) and [nalgebra](https://nalgebra.org) you can run 

`cargo bench -j8 -p tests --features tests/versus`


# Dot product benchmarks
## Benchmark 1
**This is a bad benchmark, and I know it. This is mostly just here as a place holder until I a) have more stuff to benchmark, and b) I actually start optimizing the code beyond the initial performance focused design.**

On my machine slas got slightly better performance when benchmarking dot products of vectors larger than 300 elements.
The performance advantage was only present when allocating inside the benchmark loop.
This is reasonable, since slas and ndarray both call blis. So the only part of slas that is faster is allocation, which is to be expected, since it is entirely statically allocated, whereas ndarray is dynamically allocated.

``` text
test vs_ndarray::ndarray::dot   ... bench:       5,719 ns/iter (+/- 1,185)
test vs_ndarray::slas::dot      ... bench:       5,455 ns/iter (+/- 755)
```
*Benchmark for 500 elements and 32-bit floats using blis as backend for both slas and ndarray.*

## Benchmark 2

After some tweaking slas now outperforms nalgebra and ndarray for vectors of 750 or more elements.
Slas gets 2x the performance of nalgebra with 10,000 elements.
nalgebra still gets about 3x performance for 100 elements and 10x for 10 elements.

I think this is because nalgebra's dot function is written in rust,
so it can do loop unrolling, as the vectors are statically allocated.

So my goal for now is to write a rust implementation of the dot product with loop unrolling and simd,
which will be used for vectors smaller than ~750 elements.

```
test versus::nalgebra::dot  ... bench:         159 ns/iter (+/- 6)
test versus::ndarray::dot   ... bench:         149 ns/iter (+/- 58)
test versus::slas::dot      ... bench:         146 ns/iter (+/- 4)
test versus::slas::dot_fast ... bench:         126 ns/iter (+/- 2)
```
*Benchmark for 750 elements and 32-bit floats using blis as backend for both slas and ndarray.*
