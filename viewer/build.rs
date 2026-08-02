use std::path::Path;

/// Debian/Ubuntu without `libxkbcommon-x11-dev` ships the versioned runtime
/// library but not the unversioned symlink the linker wants when building
/// gpui. Shim it into OUT_DIR instead of requiring a system package.
fn main() {
    let lib_dirs = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"];
    for dir in lib_dirs {
        let versioned = Path::new(dir).join("libxkbcommon-x11.so.0");
        let unversioned = Path::new(dir).join("libxkbcommon-x11.so");
        if versioned.exists() && !unversioned.exists() {
            let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
            let shim_dir = Path::new(&out_dir).join("link-shims");
            std::fs::create_dir_all(&shim_dir).expect("create link-shim directory");
            let shim = shim_dir.join("libxkbcommon-x11.so");
            if !shim.exists() {
                std::os::unix::fs::symlink(&versioned, &shim).expect("create link-shim symlink");
            }
            println!("cargo:rustc-link-search=native={}", shim_dir.display());
            break;
        }
    }
}
