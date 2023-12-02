use tract_onnx::prelude::*;

fn main() {
    if let Ok(model) = tract_onnx::onnx().proto_model_for_path("./nn/nn.onnx") {
        println!("{:?}", model)
    }
}
