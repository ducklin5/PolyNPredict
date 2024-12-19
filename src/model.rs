use burn::{
    config::Config, module::Module, nn::{
        loss::{MseLoss, Reduction}, lstm::{Lstm, LstmConfig}, Linear, LinearConfig
    }, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}, train::{RegressionOutput, TrainOutput, TrainStep, ValidStep}
};

use log::info;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    lstm_cell: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 16)]
    hidden_size: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            lstm_cell: LstmConfig::new(1, self.hidden_size, true).init(device),
            linear1: LinearConfig::new(self.hidden_size, self.hidden_size).with_bias(true).init(device),
            linear2: LinearConfig::new(self.hidden_size, 1).with_bias(true).init(device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BatchedData<B: Backend> {
    pub input_seq: Tensor<B, 3>,
    pub targets: Tensor<B, 2>,
}


impl<B: Backend> Model<B> {
    // input shape: (batch_size, seq_len, input_size)
    pub fn forward(&self, batched_input: Tensor<B,3>) -> Tensor<B, 2> {
        info!("batched_input shape: {:?}", batched_input.shape().dims::<3>());
        let (_, state) = self.lstm_cell.forward(batched_input, None);
        info!("last hidden shape: {:?}", state.hidden);
        let mid = self.linear1.forward(state.hidden);
        let result = self.linear2.forward(mid);


        result
    }

    pub fn compute_loss(&self, batched_data: BatchedData<B>) ->  RegressionOutput<B> {
        let BatchedData { input_seq, targets } = batched_data;
        let output = self.forward(input_seq);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Sum);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<BatchedData<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: BatchedData<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.compute_loss(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BatchedData<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: BatchedData<B>) -> RegressionOutput<B> {
        self.compute_loss(batch)
    }
}
