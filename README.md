# lynn

Linear modeling primitives and reinforcement learning algorithms in Go.

## Primitives

### Linear

A linear model (input vector -> output scalar).

```go
type Linear struct {
	Weights   []float64
	Bias      float64
	LearnRate float64
}
```

- `New(d int, learnRate float64) *Linear`
    - `d`: number of input dimensions
- `(*Linear).Feed(xs []float64) float64`: get the output of the model.
- `(*Linear).Step(gs []float64, step float64) *Linear`: perform a gradient ascent update.
    - `gs`: gradient
    - `step`: coefficient of the gradient
- `(*Linear).Regularize(strength, l1Mix float64) *Linear`: apply L1/L2 regularization when `.Step` is called.

***Note**: Remember to normalize or standardize your input features.*

### LinearGroup

A group of Linears (input vector -> output vector).

```go
type LinearGroup struct {
	K     int
	Units []*Linear
}
```

- `NewLinearGroup(k, d int, learnRate float64) *LinearGroup`
    - `k`: number of output dimensions
    - `d`: number of input dimensions
- `(*LinearGroup).Feed(xs []float64) []float64`: get the output of the model.
- `(*LinearGroup).Step(gs, unitGs []float64, step float64) *LinearGroup`: perform a gradient ascent update.
    - `gs`:the gradient
    - `unitGs`: unique coefficients of the gradient per Unit
    - `step`: coefficient of the gradient
- `(*LinearGroup).Regularize(strength, l1Mix float64) *LinearGroup`: apply L1/L2 regularization.

## Logit Functions

Two functions to create logit models from a Linear or LinearGroup.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) []float64`

## Reinforcement Learning

### RL

A vanilla REINFORCE-style training shell.

- `NewRL(policy *LinearGroup, discountRate float64) *RL`
    - Use `1` for `discountRate` if all future rewards matter equally, `0` if only immediate rewards matter, and anything in between. The normal range is `0.9` to `0.999`.
- `(*RL).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*RL).Reward(reward float64) *RL`: update the policy based on the reward, and empty the replay buffer.
- `(*RL).Regularize(strength float64) *RL`: apply entropy regularization.

### A2C

An Advantage Actor-Critic (A2C) training shell.

- `NewA2C(actor *RL, critic *Linear) *A2C`
    - The actor's policy and the critic should have the same number of input dimensions `d`. The actor's learning rate should be higher than the critic's.
- `(*A2C).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*A2C).Reward(reward float64)`: assign a reward to the most recent action.
    - If the environment provides dense rewards, assign a reward to every action. Otherwise, a single reward at the end of an episode is enough.
- `(*A2C).Finish()`: mark the most recent transition as the end of an episode.
    - This is only necessary if completing multiple episodes within one learning batch. The last reward in a batch will always be treated as the end of an episode.
- `(*A2C).Learn()`: update the actor's policy and the critic based on all rewards, and empty the replay buffer.

## Examples

### Logistic Regression

```go
model := lynn.New(3)
solver := lynn.NewGradientSolver(1e-3)

for _ = range 50 {
	for i, xs := range inputs {
		prediction := lynn.Sigmoid(model.Feed(xs))
		model.Step(xs, outputs[i]-prediction) // gradient ascent
	}
}
```

## To-Do

- [ ] Add Soft Actor-Critic (SAC)
- [ ] Add the Adam optimizer
- [ ] Add simple Q-learner?
