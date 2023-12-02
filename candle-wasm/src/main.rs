// include!("nn.onnx");
const FILE: &'static [u8] = include_bytes!("./nn.onnx");

fn main() {
    // let model = candle_onnx::read_file("./src/nn.onnx");
    // let model: candle_onnx::onnx::ModelProto =
    // protobuf::Message::parse_from_bytes(FILE.as_ref()).unwrap();
    // candle_onnx::onnx::ModelProto::from(FILE);

    // println!("{:?}", model)
}
