mod model;

use burn::tensor;
use burn_ndarray::NdArray;
use model::nn::Model;

fn main() {
    let model: Model<NdArray<f32>> = Model::new();
    let input = tensor::Tensor::<NdArray<f32>, 2>::from_data([[2.]]);
    let output = model.forward(input);
    let o = output.into_data();

    println!("{:?}", o.value);
}
