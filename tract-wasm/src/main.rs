use tract_onnx::prelude::*;

fn main() {
    let model = tract_onnx::onnx()
        .proto_model_for_path("./src/nn.onnx")
        .unwrap();
    // println!("{:?}", model);
    let info = model.training_info;
    println!("{:?}", info);
}
