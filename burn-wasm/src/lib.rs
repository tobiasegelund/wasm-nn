mod model;

use burn::{
    nn::loss::CrossEntropyLoss,
    optim::AdamConfig,
    optim::GradientsParams,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use burn_ndarray::NdArray;
use model::nn::Model;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn inference() -> Vec<f32> {
    let model: Model<NdArray<f32>> = Model::new();
    let input = Tensor::<NdArray<f32>, 2>::from_data([[2.]]);
    let output = model.forward(input);

    output.into_data().value
}

// #[wasm_bindgen]
// pub fn train() -> Vec<f32> {
//     let mut model: Model<NdArray<f32>> = Model::new();
//     let mut optim = AdamConfig::new().init();

//     let input = Tensor::<NdArray<f32>, 2>::from_floats([[2.]]);
//     let output = model.forward(input);
//     let targets: Tensor<NdArray, 1, Int> = Tensor::from_ints([1]);

//     let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets);
//     let grads = loss.backward();
//     let grads = GradientsParams::from_grads(grads, &model);

//     model = optim.step(0.01, model, grads);

//     // Output after one epoch
//     let input = Tensor::<NdArray<f32>, 2>::from_data([[2.]]);
//     let output = model.forward(input);
//     output.into_data().value
// }

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
