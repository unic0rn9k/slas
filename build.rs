use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let slas_env_vars = [("BLAS_IN_DOT_IF_LEN_GE", "750")];

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("config.rs");
    let mut f = File::create(&dest_path).unwrap();

    for (var, default_value) in slas_env_vars {
        let value = env::var(&format!("SLAS_{}", var)).unwrap_or(default_value.to_string());
        f.write_all(
            format!(
                "
                /// value of environments variable `SLAS_{var}` during build.
                /// Default value is `{default_value:?}`.
                pub const {var}: usize = {value};
             "
            )
            .as_bytes(),
        )
        .unwrap();
    }
}
