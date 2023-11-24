use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn run(path: &str) -> Vec<f32> {
    console_log!("{}", path);
    let model = tract_onnx::onnx()
        .model_for_path(path) // "./nn/nn.onnx"
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
