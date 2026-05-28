# lynn

Linear modeling primitives and reinforcement learning algorithms in Go.

## Primitives

### Unit

The simplest linear model -- multiple inputs, one output.

```go
type Unit struct {
	Weights   []float64
	Bias      float64
	LearnRate float64
}
```

- `New(n int, learnRate float64) *Unit`: create a new Unit with `n` input dimension
- `(*Unit).Feed(xs []float64) float64`: get the output of the model
- `(*Unit).Step(gs []float64, step float64)`: perform a gradient ascent update where `gs` is the gradient and `step` is the coefficient of the gradient

***Note**: Remember to normalize or standardize your input features.*

### Layer

A parellel group of Units -- multiple inputs, multiple outputs.

```go
type Layer struct {
	K     int
	Units []*Unit
}
```

- `NewLayer(k, n int, learnRate float64) *Layer`: create a new Layer with `k` output dimensions and `n` input dimensions
- `(*Layer).Feed(xs []float64) []float64`: get the output of the model
- `(*Layer).Step(gs, unitGs []float64, step float64)`: perform a gradient ascent update where `gs` is the gradient, `unitGs` are the coefficients of the gradient for each Unit, and `step` is the coefficient of the gradient

## Logits

Two functions are provided to create logit models from a Unit or Layer.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) []float64`

## Reinforcement Learning

### RL

A vanilla REINFORCE-style training shell.

- `NewRL(policy *Layer, discountRate float64) *RL`
- `(*RL).Act(state []float64) int`: get an action in the range `[0, k - 1]`
- `(*RL).Reward(reward float64)`: apply a time-discounted reward for all actions taken since the last reward

***Note**: Discount rate is gamma. Use `1` for no discounting.*

### A2C

An Advantage Actor-Critic (A2C) training shell.

- `NewA2C(actor *RL, critic *Unit) *A2C`
- `(*A2C).Act(state []float64) int`: get an action in the range `[0, k - 1]`
- `(*A2C).Reward(reward float64)`: apply a time-discounted reward for all actions taken since the last reward

***Note**: The actor's policy and the critic should have the same number of input dimensions `n`. The actor's learning rate should be higher than the critic's.*

## Examples

### Linear Regression

```go
u := lynn.New(3, 1e-3)

for _ = range 500 {
	for i, xs := range inputs {
		prediction := u.Feed(xs)
		u.Step(xs, outputs[i]-prediction) // gradient ascent
	}
}
```

## To-Do

- [ ] Add temporal differencing to A2C
- [ ] Add simple Q-learner
- [ ] Add the Adam optimizer
