mod model;

use burn::tensor;
use burn_ndarray::NdArray;
use model::nn::Model;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn inference() -> Vec<f32> {
    let model: Model<NdArray<f32>> = Model::new();
    let input = tensor::Tensor::<NdArray<f32>, 2>::from_data([[2.]]);
    let output = model.forward(input);

    output.into_data().value
}

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
