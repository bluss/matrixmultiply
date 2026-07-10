fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let _ = match autocfg::AutoCfg::new() {
        Err(err) => {
            println!("cargo:warning={}", err);
            return;
        },
        Ok(ac) => ac,
    };
}
