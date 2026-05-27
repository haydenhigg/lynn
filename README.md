# lynn

Optimized linear models in Go.

## Primitives

### Unit

The simplest linear model -- multiple inputs, one output.

- `New(n int, learningRate float64) *Unit`
- `(*Unit).Feed(xs []float64) float64`: get the raw output of the model
- `(*Unit).Step(delta float64, xs []float64)`: ascend the gradient where `delta` is the coefficient of the gradients

***Note**: Remember to normalize or standardize your input features.*

### Block

A layer of Units -- multiple inputs, multiple outputs.

- `NewBlock(k, n int, learningRate float64) *Block` (k is the number of outputs, n is the number of inputs)
- `(*Block).Feed(xs []float64) float64`: get the raw output of the model
- `(*Block).Step(blockDelta float64, deltas float64, xs []float64)`: ascend the gradient where `blockDelta` is the coefficient of all gradients and `deltas` are the coefficients of the gradients for each Unit

## Logit Transformations

Two functions are provided to create logit models from Units or Blocks.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) float64`

## Training

### RL

An advantage actor critic (A2C) harness for a Block.

- `NewRL(k, n int, actorLearningRate, criticLearningRate float64) *RL`  (k is the number of outputs, n is the number of inputs)
- `(*RL).Act(state []float64) int`: get an action (0-`K`)
- `(*RL).Reward(reward float64)`: apply a reward to all actions taken since the last reward

***Note**: The critic's learning rate should be lower than the actor's.*

***Note**: Input regularization is still necessary.*

## To-Do

- [ ] Add vanilla RL in addition to A2C
- [ ] Set up a test with chrys/anthemum
