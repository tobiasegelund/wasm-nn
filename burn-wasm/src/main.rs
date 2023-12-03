mod model;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    nn::loss::CrossEntropyLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Int, Tensor},
};
use burn_ndarray::NdArray;
use model::nn::Model;

fn train<B: AutodiffBackend>() {
    let mut model = Model::<B>::new();
    let mut optim = AdamConfig::new().init();

    let input: Tensor<B, 2> = Tensor::<B, 2>::from_floats([[2.]]);
    let output = model.forward(input);
    println!("{:?}", output.clone().into_data());

    let targets: Tensor<B, 1, Int> = Tensor::from_ints([1]);

    let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets);
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    model = optim.step(0.01, model, grads);

    // Output after one epoch
    let input = Tensor::<B, 2>::from_floats([[2.]]);
    let output = model.forward(input);
    println!("{:?}", output.clone().into_data());
}

fn main() {
    train::<Autodiff<Wgpu>>();
}
