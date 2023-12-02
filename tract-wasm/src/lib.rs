use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

const FILE: &'static [u8] = include_bytes!("./nn.onnx");

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn inference() -> Vec<f32> {
    let input = tract_ndarray::arr2(&[[1.0f32]]).into_tensor();
    if let Ok(model) = tract_onnx::onnx().model_for_read(&mut FILE.as_ref()) {
        let output = model
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap()
            .run(tvec![input.into()])
            .unwrap();
        output[0].to_owned().as_slice().unwrap().into()
    } else {
        vec![0.0]
    }
}

#[wasm_bindgen]
pub fn train() -> i32 {
    if let Ok(model) = tract_onnx::onnx().proto_model_for_read(&mut FILE.as_ref()) {
        // model
    }
    10
}

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
