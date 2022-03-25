use std::mem::size_of;

const LANES_8: bool = cfg!(any(
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma"
));

const LANES_4: bool = cfg!(any(
    target_feature = "sse",
    target_feature = "sse2",
    target_feature = "sse3",
    target_feature = "ssse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "sse4a",
));

const LANES_16: bool = cfg!(target_feature = "avx512");

/// Available SIMD lanes on CPU architecutre compiled on for the f32 type.
/// So if you want to make a SIMD vector of f64 you would want to use LANES/2.
/// Will be 0 if SIMD is not available.
pub const MAX: usize = if LANES_16 {
    16
} else if LANES_8 {
    8
} else if LANES_4 {
    4
} else {
    0
};

/// Returns the number of lanes, for a given type, that can fit into a SIMD vektor on the architecture compiled on.
pub const fn max_for_type<T>() -> usize {
    MAX / (size_of::<T>() / size_of::<f32>())
}
