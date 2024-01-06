// include!("nn.onnx");
use candle_core::{Device, Tensor};
use prost::Message;

use std::collections::HashMap;

const FILE: &'static [u8] = include_bytes!("./nn.onnx");

fn main() {
    let model = candle_onnx::onnx::ModelProto::decode(FILE.as_ref()).unwrap();
    // let graph = model.graph.as_ref().unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::new(&[2f32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap(),
    );

    let output = candle_onnx::simple_eval(&model, inputs);

    // println!("{:?}", model);
    println!("{:?}", output); // .unwrap().remove("output")
}
