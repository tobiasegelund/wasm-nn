use std::collections::HashMap;

use candle_core::{Device, Tensor};
use prost::Message;
use wasm_bindgen::prelude::*;

const FILE: &'static [u8] = include_bytes!("./nn.onnx");

#[wasm_bindgen]
pub fn inference() -> Vec<f32> {
    let model = candle_onnx::onnx::ModelProto::decode(FILE.as_ref()).unwrap();
    // let model = candle_onnx::read_file("./nn.onnx").unwrap();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::new(&[2f32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap(),
    );

    let output = candle_onnx::simple_eval(&model, inputs);
    output.unwrap().remove("output").unwrap().to_vec1().unwrap()
}
