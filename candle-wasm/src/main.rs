// include!("nn.onnx");
use prost::Message;

const FILE: &'static [u8] = include_bytes!("./nn.onnx");

fn main() {
    let model = candle_onnx::onnx::ModelProto::decode(FILE.as_ref()).unwrap();

    println!("{:?}", model)
}
