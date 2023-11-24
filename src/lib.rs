use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run() -> Vec<f32> {
    println!("enter");
    let model = tract_onnx::onnx()
        .model_for_path("./nn/nn.onnx")
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    let input = tract_ndarray::arr2(&[[1.0f32]]).into_tensor();
    let output = model.run(tvec![input.into()]).unwrap();

    output[0].to_owned().as_slice().unwrap().into()
}

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
