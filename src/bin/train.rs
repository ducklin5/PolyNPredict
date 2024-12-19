use burn::{
    backend::{Autodiff, Wgpu},
    data::{dataloader::batcher::Batcher, dataset::Dataset, dataloader::DataLoaderBuilder},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::LossMetric,
        LearnerBuilder,
    },
};
use burndemo::model::{BatchedData, ModelConfig};
use itertools::iproduct;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 3)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 2e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

// Dataset wrapper for generic Vec<T>
struct VecDataset<T>(Vec<T>);

impl<T> Dataset<T> for VecDataset<T>
where
    T: Copy + Sync + Send,
{
    fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl <T: Clone> VecDataset<T> {
    pub fn rand_split(&self, ratio: f64) -> (Self, Self) {
        let mut rng = thread_rng();
        let mut data = self.0.clone();
        data.shuffle(&mut rng);

        let n = self.0.len();
        let n1 = (n as f64 * ratio).round() as usize;
        let (left, right) = data.split_at(n1);
        (VecDataset(left.to_vec()), VecDataset(right.to_vec()))
    }
}

#[derive(Clone)]
pub struct VecDSBatcher<B: Backend>(B::Device);

type DataEntry<const D: usize> = ([f64; D], f64);

impl<B: Backend, const D: usize> Batcher<DataEntry<D>, BatchedData<B>> for VecDSBatcher<B> {
    fn batch(&self, items: Vec<DataEntry<D>>) -> BatchedData<B> {
        let batch_size = items.len();
        let (inputs, targets): (Vec<[f64; D]>, Vec<f64>) = items.into_iter().unzip();
        let inputs = inputs.into_iter().flat_map(|seq| seq).collect::<Vec<f64>>();

        let input_seq =
            Tensor::<B, 1>::from_floats(inputs.as_slice(), &self.0).reshape([batch_size, D, 1]);
        let targets =
            Tensor::<B, 1>::from_floats(targets.as_slice(), &self.0).reshape([batch_size, 1]);
        BatchedData { input_seq, targets }
    }
}

pub fn generate_poly2_entry<const D: usize>(m2: f64, m1: f64, b:f64, sx: f64, dx:f64) -> DataEntry<D> {
    let compute_ith = |i: usize| {
        let iflt = i as f64;
        let x = sx + iflt * dx;
        m2 * (x).powi(2) + m1 * x + b
    };
    let seq = (0..D).map(compute_ith).collect::<Vec<f64>>().try_into().unwrap();
    let next = compute_ith(D);
    (seq, next)
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );
    
    let m2_values = (0..1).map(|i| 0.0 * i as f64);
    let m1_values = (-15..=15).map(|i| 1.3 * i as f64);
    let b_values = (-5..=5).map(|i| 2.0 * i as f64);
    let sx_values = (-5..=5).map(|i| 2.4 * i as f64);
    let dx_values = (-3..=3).map(|i| 0.5 * i as f64);
    
    let coef_combo = iproduct!(m2_values, m1_values, b_values, sx_values, dx_values);
    let data: VecDataset<DataEntry<5>> = VecDataset(
        coef_combo.map(|(m2, m1, b, sx, dx)| generate_poly2_entry::<5>(m2, m1, b, sx, dx)).collect()
    );

    let (train_data, valid_data) = data.rand_split(0.6);
   
    let batcher_train = VecDSBatcher::<B>(device.clone());
    let batcher_valid = VecDSBatcher::<B::InnerBackend>(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_data);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_data);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/burndemo";

    // Train the model
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}
