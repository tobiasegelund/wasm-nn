use burn_import::onnx::ModelGen;

// const FILE: &'static [u8] = include_bytes!("./nn.onnx");

fn main() {
    // Generate Rust code from the ONNX model file
    ModelGen::new()
        .input("src/model/nn.onnx")
        .out_dir("model/")
        .run_from_script();
}
