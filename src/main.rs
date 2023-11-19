use tract_ndarray::arr2;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("./nn/nn.onnx")?
        .into_optimized()?
        .into_runnable()?;

    let input = arr2(&[[1.0f32]]).into_tensor();
    let output = model.run(tvec![input.into()])?;
    println!("{}", output[0].to_array_view::<f32>()?);

    // let input2: Tensor = tract_ndarray::Array2::from_shape_vec((1, 1), vec![2.0f32])?.into();
    // let output2 = model.run(tvec![input2.into()])?;
    // println!("{:?}", output2);

    Ok(())
}
