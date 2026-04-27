use std::path::Path;

fn main() {
	println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
	println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

	if !cfg!(target_family = "unix") || std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some()
	{
		return;
	}

	let config = pyo3_build_config::get();
	if !config.shared {
		return;
	};

	let Some(libdir) = config.lib_dir.as_deref() else {
		return;
	};

	if libdir.is_empty() || !Path::new(libdir).is_dir() {
		return;
	}

	println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
