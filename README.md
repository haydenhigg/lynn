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

- `New(d int, learnRate float64) *Unit`: create a new Unit with `d` input dimension
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

- `NewLayer(k, d int, learnRate float64) *Layer`: create a new Layer with `k` output dimensions and `d` input dimensions.
- `(*Layer).Feed(xs []float64) []float64`: get the output of the model.
- `(*Layer).Step(gs, unitGs []float64, step float64)`: perform a gradient ascent update where `gs` is the gradient, `unitGs` are the coefficients of the gradient for each Unit, and `step` is the coefficient of the gradient.

## Logits

Two functions are provided to create logit models from a Unit or Layer.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) []float64`

## Reinforcement Learning

### RL

A vanilla REINFORCE-style training shell.

- `NewRL(policy *Layer, discountRate float64) *RL`
    - Discount rate is gamma. Use `1` for no discounting and `0` for full discounting (myopic). The normal range is `0.9` to `0.999`.
- `(*RL).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*RL).Reward(reward float64)`: update the policy based on the reward, and empty the replay buffer.

### A2C

An Advantage Actor-Critic (A2C) training shell.

- `NewA2C(actor *RL, critic *Unit) *A2C`
    - The actor's policy and the critic should have the same number of input dimensions `d`. The actor's learning rate should be higher than the critic's.
- `(*A2C).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*A2C).Reward(reward float64)`: assign a reward to the most recent action.
    - If the environment provides dense rewards, assign a reward to every action. Otherwise, a single reward at the end of an episode is enough.
- `(*A2C).Done()`: mark the most recent transition as the end of an episode.
    - This is only necessary if completing multiple episodes within one learning batch. The last reward in a batch will always be treated as the end of an episode.
- `(*A2C).Learn()`: update the actor's policy and the critic based on all rewards, and empty the replay buffer.

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

- [ ] Add entropy regularization
- [ ] Add simple Q-learner
- [ ] Add the Adam optimizer
