// Generated from ONNX "./src/model/nn.onnx" by burn-import
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    phantom: core::marker::PhantomData<B>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("./src/model/nn")
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into())
            .expect("Record file to exist.");
        Self::new_with(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new_with(record: ModelRecord<B>) -> Self {
        let linear1 = LinearConfig::new(1, 50)
            .with_bias(true)
            .init_with(record.linear1);
        let linear2 = LinearConfig::new(50, 6)
            .with_bias(true)
            .init_with(record.linear2);
        Self {
            linear1,
            linear2,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        let linear1 = LinearConfig::new(1, 50).with_bias(true).init();
        let linear2 = LinearConfig::new(50, 6).with_bias(true).init();
        Self {
            linear1,
            linear2,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 2>) -> Tensor<B, 2> {
        let relu1_out1 = burn::tensor::activation::relu(input1);
        let linear1_out1 = self.linear1.forward(relu1_out1);
        let linear2_out1 = self.linear2.forward(linear1_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(linear2_out1, 1);
        softmax1_out1
    }
}
