// include!("nn.onnx");
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

// Below fails due to `unsupported op_type Gemm for op NodeProto`
// const FILE: &'static [u8] = include_bytes!("./nn.onnx");
// use prost::Message;
// use std::collections::HashMap;

// fn load_model() {
//     let model = candle_onnx::onnx::ModelProto::decode(FILE.as_ref()).unwrap();
//     // let graph = model.graph.as_ref().unwrap();

//     let mut inputs = HashMap::new();
//     inputs.insert(
//         "input".to_string(),
//         Tensor::new(&[2f32], &Device::Cpu)
//             .unwrap()
//             .reshape((1, 1))
//             .unwrap(),
//     );

//     let output = candle_onnx::simple_eval(&model, inputs);

//     // println!("{:?}", model);
//     println!("{:?}", output); // .unwrap().remove("output")
// }

struct Model {
    ln1: Linear,
    ln2: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(1, 50, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(50, 6, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

fn main() {
    let dev = Device::Cpu;

    let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 1., 2., 3., 4.], (9, 1), &dev).unwrap();
    let y = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 2, 3, 4, 5], 9, &dev).unwrap();

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(vs.clone()).unwrap();

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), 0.01).unwrap();

    for epoch in 1..5 {
        let logits = model.forward(&x).unwrap();
        let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
        let loss = loss::nll(&log_sm, &y).unwrap();
        sgd.backward_step(&loss).unwrap();
        println!(
            "{epoch:4} train loss: {:8.5}",
            loss.to_scalar::<f32>().unwrap(),
        );
    }
}
