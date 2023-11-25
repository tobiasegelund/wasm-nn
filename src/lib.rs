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
    // "./nn/nn.onnx"
    console_log!("{}", path);
    let input = tract_ndarray::arr2(&[[1.0f32]]).into_tensor();
    if let Ok(model) = tract_onnx::onnx().model_for_path(path) {
        let output = model
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap()
            .run(tvec![input.into()])
            .unwrap();
        output[0].to_owned().as_slice().unwrap().into()
    } else {
        vec![10.0]
    }
}

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
