# lynn

Linear modeling primitives and reinforcement learning algorithms in Go.

## Primitives

### Linear

A linear model (input vector -> output scalar).

```go
type Linear struct {
	D       int
	Weights []float64
	Bias    float64
}
```

- `New(d int) *Linear`
    - `d`: number of input dimensions
- `(*Linear).Randomize(mu, sigma float64) *Linear`: initialize weights and bias from a normal distribution.
- `(*Linear).Feed(xs []float64) float64`: get the output of the model.
- `(*Linear).Step(xs []float64, scale float64) *Linear`: perform a gradient ascent update.
    - `xs`: input vector/gradient direction
    - `scale`: coefficient of the update

***Note**: Remember to normalize or standardize your input features.*

### LinearGroup

A group of Linears (input vector -> output vector).

```go
type LinearGroup struct {
	K     int
	Units []*Linear
}
```

- `NewLinearGroup(k, d int) *LinearGroup`
    - `k`: number of output dimensions
    - `d`: number of input dimensions
- `(*LinearGroup).Feed(xs []float64) []float64`: get the output of the model.
- `(*LinearGroup).Step(xs, us []float64, scale float64) *LinearGroup`: perform a gradient ascent update.
    - `xs`: input vector/gradient direction
    - `us`: unique coefficients of the update per unit
    - `scale`: coefficient of the update

### GradientSolver

A simple gradient solver for training linear models, with optional L1/L2 regularization.

```go
type GradientSolver struct {
	LearnRate float64
	L1Penalty float64
	L2Penalty float64
}
```

- `NewGradientSolver(learnRate float64) *GradientSolver`
- `(*GradientSolver).Lasso(strength float64) *GradientSolver`: apply L1 regularization.
- `(*GradientSolver).Ridge(strength float64) *GradientSolver`: apply L2 regularization.
- `(*GradientSolver).ElasticNet(strength, l1Mix float64) *GradientSolver`: apply mixed L1/L2 regularization.
- `(*GradientSolver).Ascend(l *Linear, xs []float64, scale float64) *GradientSolver`: perform a gradient ascent update.
- `(*GradientSolver).Descend(l *Linear, xs []float64, scale float64) *GradientSolver`: perform a gradient descent update.

## Logit Functions

Two functions to create logit models from a Linear or LinearGroup.

- `Sigmoid(z float64) float64`
- `Softmax(zs []float64) []float64`

## Reinforcement Learning

### Transition

A replay-buffer transition used by reinforcement learning trainers.

```go
type Transition struct {
	State           []float64
	ActionGradient  []float64
	EntropyGradient []float64
	Reward          float64
	Done            bool
}
```

### RL

A vanilla REINFORCE-style training shell.

```go
type RL struct {
	Policy          *LinearGroup
	LearnRate       float64
	DiscountRate    float64
	ExplorePressure float64
	Trajectory      []Transition
}
```

- `NewRL(policy *LinearGroup, learnRate, discountRate float64) *RL`
    - `policy`: action policy model
    - `learnRate`: policy learning rate
    - `discountRate`: use `1` if all future rewards matter equally, `0` if only immediate rewards matter, and anything in between. The normal range is `0.9` to `0.999`.
- `(*RL).Regularize(strength float64) *RL`: apply entropy regularization.
- `(*RL).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*RL).Reward(reward float64) *RL`: update the policy based on the reward, and empty the replay buffer.

### A2C

An Advantage Actor-Critic (A2C) training shell.

```go
type A2C struct {
	Actor     *RL
	Critic    *Linear
	LearnRate float64
}
```

- `NewA2C(actor *RL, learnRate float64) *A2C`
    - `actor`: policy trainer
    - `learnRate`: critic learning rate
- `(*A2C).Act(state []float64) int`: get an action in the range `[0, k - 1]`.
- `(*A2C).Reward(reward float64) *A2C`: assign a reward to the most recent action.
    - If the environment provides dense rewards, assign a reward to every action. Otherwise, a single reward at the end of an episode is enough.
- `(*A2C).Finish() *A2C`: mark the most recent transition as the end of an episode.
    - This is only necessary if completing multiple episodes within one learning batch. The last reward in a batch will always be treated as the end of an episode, even if this is not called.
- `(*A2C).Learn() *A2C`: update the actor's policy and the critic based on all rewards, and empty the replay buffer.

## Examples

### Logistic Regression

```go
model := lynn.New(3)
solver := lynn.NewGradientSolver(1e-3)

for _ = range 50 {
	for i, xs := range inputs {
		prediction := lynn.Sigmoid(model.Feed(xs))
		solver.Descend(model, xs, prediction-outputs[i])
	}
}
```

## To-Do

- [ ] Add OLS solver
- [ ] Add Adam solver
- [ ] Add Soft Actor-Critic (SAC)
