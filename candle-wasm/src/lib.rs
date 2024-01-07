use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use prost::Message;
use wasm_bindgen::prelude::*;

const FILE: &'static [u8] = include_bytes!("./nn.onnx");
const DEVICE: Device = Device::Cpu;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct Model {
    ln1: Linear,
    ln2: Linear,

    varmap: VarMap,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);

        let ln1 = candle_nn::linear(1, 50, vs.pp("ln1")).unwrap();
        let ln2 = candle_nn::linear(50, 6, vs.pp("ln2")).unwrap();

        Self { ln1, ln2, varmap }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln1.forward(x)?;
        let x = x.relu()?;
        self.ln2.forward(&x)
    }

    pub fn train(&self, input: f32, label: u8) -> std::result::Result<Vec<f32>, JsError> {
        let x = Tensor::from_vec(vec![input], (1, 1), &DEVICE).unwrap();
        let y = Tensor::from_vec(vec![label], 1, &DEVICE).unwrap();

        let mut sgd = candle_nn::SGD::new(self.varmap.all_vars(), 0.01).unwrap();

        let logits = self.forward(&x).unwrap();
        let log_sm = ops::softmax(&logits, D::Minus1).unwrap();
        let loss = loss::cross_entropy(&log_sm, &y).unwrap();
        console_log!("LOSS: {}", loss.to_scalar::<f32>().unwrap());
        sgd.backward_step(&loss).unwrap();
        let result = log_sm.clone().to_vec2::<f32>().unwrap();

        Ok(result.get(0).unwrap().to_owned())
    }
}

#[wasm_bindgen]
pub fn train(input: f32, label: u8) -> Vec<f32> {
    let x = Tensor::from_vec(vec![input], (1, 1), &DEVICE).unwrap();
    let y = Tensor::from_vec(vec![label], 1, &DEVICE).unwrap();

    let varmap = VarMap::new();
    let model = Model::new();

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), 0.01).unwrap();

    let logits = model.forward(&x).unwrap();
    let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
    let loss = loss::nll(&log_sm, &y).unwrap();
    sgd.backward_step(&loss).unwrap();
    let result = log_sm.clone().to_vec2::<f32>().unwrap();

    result.get(0).unwrap().to_owned()
}

#[wasm_bindgen]
pub fn inference_onnx(input: f32) {
    let model = candle_onnx::onnx::ModelProto::decode(FILE.as_ref()).unwrap();
    // let model = candle_onnx::read_file("./nn.onnx").unwrap();

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::new(&[input], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap(),
    );

    let output = candle_onnx::simple_eval(&model, inputs);
    console_log!("{:?}", output)
}
