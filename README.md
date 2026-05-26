# lynn

Linear models, trainable by SGD or A2C reinforcement learning.

## Models

### `Bernoulli`

The simplest linear model. Multiple inputs, one output.

- `NewBernoulli(n int, learningRate float64) *Bernoulli`
- `(*Bernoulli).Feed(xs []float64) float64`: get the raw output of the model
- `(*Bernoulli).Prob(xs []float64) float64`: get the sigmoid of the output of the model
- `(*Bernoulli).Step(d float64, xs []float64)`: ascend the gradient where `d` is the coefficient of the gradient

## Training

### `RL`

An advantage actor critic (A2C) harness for a `Bernoulli` model.
