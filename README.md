# lynn

Optimized linear models in Go.

## Models

### Bernoulli

The simplest linear model -- multiple inputs, one output.

- `NewBernoulli(n int, learningRate float64) *Bernoulli`
- `(*Bernoulli).Feed(xs []float64) float64`: get the raw output of the model
- `(*Bernoulli).Prob(xs []float64) float64`: get the sigmoid of the output of the model
- `(*Bernoulli).Step(d float64, xs []float64)`: ascend the gradient where `d` is the coefficient of the gradient

***Note**: Remember to normalize or standardize your input features.*

## Training

### RL

An advantage actor critic (A2C) harness for a `*Bernoulli` model.

- `NewRL(actor, critic *Bernoulli)`
- `(*RL).Act(state []float64) int`: get an action (0 or 1)
- `(*RL).Reward(reward float64)`: apply a reward to all actions taken since the last reward

***Note**: The critic's learning rate should be lower than the actor's.*
