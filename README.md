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

- `New(d int, learnRate float64) *Linear`: create a new Linear with `d` input dimension.
- `(*Linear).Feed(xs []float64) float64`: get the output of the model.
- `(*Linear).Step(gs []float64, step float64)`: perform a gradient ascent update where `gs` is the gradient and `step` is the coefficient of the gradient.
- `(*Linear).Regularize(strength, l1Mix float64) float64`: apply L1/L2 regularization when `.Step` is called.

***Note**: Remember to normalize or standardize your input features.*

### LinearGroup

A group of Linears (input vector -> output vector).

```go
type LinearGroup struct {
	K     int
	Units []*Linear
}
```

- `NewLinearGroup(k, d int, learnRate float64) *LinearGroup`: create a new LinearGroup with `k` output dimensions and `d` input dimensions.
- `(*LinearGroup).Feed(xs []float64) []float64`: get the output of the model.
- `(*LinearGroup).Step(gs, unitGs []float64, step float64)`: perform a gradient ascent update where `gs` is the gradient, `unitGs` are the coefficients of the gradient for each Linear, and `step` is the coefficient of the gradient.
- `(*LinearGroup).Regularize(strength, l1Mix float64) float64`: apply L1/L2 regularization when `.Step` is called.

## Logit Functions

Two functions to create logit models from a Linear or LinearGroup.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) []float64`

## Reinforcement Learning

### RL

A vanilla REINFORCE-style training shell.

- `NewRL(policy *LinearGroup, gamma, beta float64) *RL`
    - `gamma` is the discounting rate. Use `1` for no discounting (all future rewards matter equally) and `0` for full discounting (only immediate rewards matter). The normal range is `0.9` to `0.999`.
    - `beta` is the exploration pressure. Use `0` for no entropy regularization.
- `(*RL).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*RL).Reward(reward float64)`: update the policy based on the reward, and empty the replay buffer.

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
l := lynn.New(3, 1e-3)
l.Regularize(0.1, 0.5)

for _ = range 50 {
	for i, xs := range inputs {
		prediction := lynn.Sigmoid(l.Feed(xs))
		l.Step(xs, outputs[i]-prediction) // gradient ascent
	}
}
```

## To-Do

- [x] Use temporal differencing in A2C
- [x] Add entropy regularization
- [ ] Add simple Q-learner
- [ ] Add the Adam optimizer
