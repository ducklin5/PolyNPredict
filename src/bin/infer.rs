use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    backend::Wgpu
};
use burndemo::model::{ModelConfig, Model};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, seq: Vec<f64>) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");
    
    let model: Model<B> = ModelConfig::new().init(&device).load_record(record);
    let input = Tensor::<B, 1>::from_floats(seq.as_slice(), &device).reshape([1, seq.len(), 1]);
    
    let predicted = model.forward(input);
    
    println!("Predicted {}", predicted);
}


fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/burndemo";

    let seq_str = std::env::args().nth(1).unwrap();
    let seq: Vec<f64> = seq_str.split(',').map(|x| x.parse().unwrap()).collect();


    // Infer the model
    infer::<MyBackend>(
        artifact_dir,
        device,
        seq,
    );
}
