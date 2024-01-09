use std::env;
use std::path::PathBuf;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    compile_source(
        "grumpkin_pippenger.cpp",
        "__BLST_PORTABLE__",
        "grumpkin_msm",
        &target_arch,
    );
    compile_source(
        "pasta_pippenger.cpp",
        "__PASTA_PORTABLE__",
        "pasta_msm",
        &target_arch,
    );

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }

    if cuda_available() {
        let mut implement_sort: bool = true;
        compile_cuda("cuda/bn254.cu", "bn256_msm_cuda", implement_sort);
        implement_sort = false;
        compile_cuda("cuda/grumpkin.cu", "grumpkin_msm_cuda", implement_sort);
        compile_cuda("cuda/pallas.cu", "pallas_msm_cuda", implement_sort);
        compile_cuda("cuda/vesta.cu", "vesta_msm_cuda", implement_sort);
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
        #[cfg(feature = "cuda")]
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
}

fn compile_source(
    file_name: &str,
    def: &str,
    output_name: &str,
    target_arch: &str,
) {
    let mut cc = cc::Build::new();
    cc.cpp(true);

    let c_src_dir = PathBuf::from("src");
    let file = c_src_dir.join(file_name);
    let cc_def = determine_cc_def(target_arch, def);

    common_build_configurations(&mut cc);
    if let Some(cc_def) = cc_def {
        cc.define(&cc_def, None);
    }
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        cc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
        cc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
        cc.include(include);
    }
    cc.file(file).compile(output_name);
}

fn common_build_configurations(cc: &mut cc::Build) {
    cc.flag_if_supported("-mno-avx")
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.define("NDEBUG", None);
    }
}

fn determine_cc_def(target_arch: &str, default_def: &str) -> Option<String> {
    assert!(
        !(cfg!(feature = "portable") && cfg!(feature = "force-adx")),
        "Cannot compile with both `portable` and `force-adx` features"
    );

    if cfg!(feature = "portable") {
        return Some(default_def.to_string());
    }

    if cfg!(feature = "force-adx") && target_arch == "x86_64" {
        return Some("__ADX__".to_string());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if target_arch == "x86_64" && std::is_x86_feature_detected!("adx") {
            return Some("__ADX__".to_string());
        }
    }

    None
}

fn cuda_available() -> bool {
    match env::var("NVCC") {
        Ok(var) => which::which(var).is_ok(),
        Err(_) => which::which("nvcc").is_ok(),
    }
}

fn compile_cuda(file_name: &str, output_name: &str, implement_sort: bool) {
    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.flag("-arch=sm_80");
    nvcc.flag("-gencode").flag("arch=compute_70,code=sm_70");
    nvcc.flag("-t0");
    #[cfg(not(target_env = "msvc"))]
    nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
    nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
    #[cfg(feature = "cuda-mobile")]
    nvcc.define("NTHREADS", "128");

    if let Some(def) = determine_cc_def(
        &env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        "__CUDA_PORTABLE__",
    ) {
        nvcc.define(&def, None);
    }

    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        nvcc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
        nvcc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
        nvcc.include(include);
    }
    if implement_sort {
        nvcc.file(file_name).compile(output_name);
    } else {
        nvcc.define("__MSM_SORT_DONT_IMPLEMENT__", None)
            .file(file_name)
            .compile(output_name);
    }
}
