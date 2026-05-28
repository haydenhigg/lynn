# lynn

Optimized linear models in Go.

## Primitives

### Unit

The simplest linear model -- multiple inputs, one output.

- `New(n int, learnRate float64) *Unit`
- `(*Unit).Feed(xs []float64) float64`: get the raw output of the model
- `(*Unit).Step(delta float64, xs []float64)`: ascend the gradient where `delta` is the coefficient of the gradients

***Note**: Remember to normalize or standardize your input features.*

### Layer

A parellel group of Units -- multiple inputs, multiple outputs.

- `NewLayer(k, n int, learnRate float64) *Layer` (k is the number of outputs, n is the number of inputs)
- `(*Layer).Feed(xs []float64) float64`: get the raw output of the model
- `(*Layer).Step(blockDelta float64, deltas float64, xs []float64)`: ascend the gradient where `blockDelta` is the coefficient of all gradients and `deltas` are the coefficients of the gradients for each Unit

## Logits

Two functions are provided to create logit models from a Unit or Layer.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) float64`

## Reinforcement Learning

### RL

A vanilla REINFORCE-style harness for a Layer.

- `NewRL(policy *Layer, discountRate float64) *RL`
- `(*RL).Act(state []float64) int`: choose an action in the range `[0, k - 1]`
- `(*RL).Reward(reward float64)`: apply a reward to all actions taken since the last reward

***Note**: Discount rate is gamma. Use `1` for no discounting.*

### A2C

An advantage actor critic (A2C) harness for a Layer.

- `NewA2C(policy *Layer, critic *Unit, discountRate float64) *A2C`
- `(*A2C).Act(state []float64) int`: choose an action in the range `[0, k - 1]`
- `(*A2C).Reward(reward float64)`: apply a reward to all actions taken since the last reward minus the expected reward from that state

***Note**: The critic and the actor should have the same number of input dimensions `n`. The critic's learning rate should be smaller than the actor's.*

## To-Do

- [ ] Add temporal differencing to A2C
- [ ] Add simple Q-learner
- [ ] Add the Adam optimizer
