# POLYN Sequence Predictor 

- NOTE: only only trained on poly 1 for now

- Program uses burn LSTM + Linear modules to predict the next number in a leanr sequnece 

- Trained on and works best with 5 preceeding inputs

- Yes.. I know this is a pointless program. I code for the LEARNINGSSSSSS

## Running

- Requires dedicated GPU

### Training

`cargo run --bin train --release`

### Inference

`cargo run --bin infer 18,6,-6,-18,-30`
