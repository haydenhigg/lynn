package lynn

import (
	"math"
	"math/rand"
)

type Transition struct {
	State       []float64
	ActionGrad  []float64
	EntropyGrad []float64
	Reward      float64
	Done        bool
}

type RL struct {
	Policy     *Layer
	Gamma      float64 // discount rate
	Beta       float64 // exploration pressure
	Trajectory []Transition
}

func NewRL(policy *Layer, gamma, beta float64) *RL {
	return &RL{policy, gamma, beta, []Transition{}}
}

func sampleAction(ps []float64) int {
	v := rand.Float64()

	for action, p := range ps {
		if v -= p; v < 0 {
			return action
		}
	}

	return len(ps) - 1
}

func (rl *RL) Act(state []float64) int {
	ps := Softmax(rl.Policy.Feed(state))
	action := sampleAction(ps)

	rl.Trajectory = append(rl.Trajectory, Transition{
		State:       state,
		ActionGrad:  actionErrors(ps, action),
		EntropyGrad: entropyErrors(ps),
	})

	return action
}

func actionErrors(ps []float64, action int) []float64 {
	errors := make([]float64, len(ps))

	for i, p := range ps {
		if action == i {
			errors[i] = 1 - p
		} else {
			errors[i] = 0 - p
		}
	}

	return errors
}

func entropy(ps []float64) float64 {
	entropy := 0.

	for _, p := range ps {
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}

	return entropy
}

func entropyErrors(ps []float64) []float64 {
	h := entropy(ps)
	errors := make([]float64, len(ps))

	for i, p := range ps {
		errors[i] = -p * (math.Log(p) + h)
	}

	return errors
}

func (rl *RL) Reward(reward float64) {
	t := len(rl.Trajectory) - 1
	discountFactor := 1.

	for i := range rl.Trajectory {
		transition := rl.Trajectory[t-i]

		rl.Policy.Step(transition.State, transition.ActionGrad, reward*discountFactor)
		rl.Policy.Step(transition.State, transition.EntropyGrad, rl.Beta)

		discountFactor *= rl.Gamma
	}

	rl.Trajectory = []Transition{}
}

type A2C struct {
	Actor  *RL
	Critic *Unit
}

func NewA2C(actor *RL, critic *Unit) *A2C {
	return &A2C{actor, critic}
}

func (a2c *A2C) Act(state []float64) int {
	return a2c.Actor.Act(state)
}

func (a2c *A2C) Reward(reward float64) {
	t := len(a2c.Actor.Trajectory) - 1
	if t >= 0 {
		a2c.Actor.Trajectory[t].Reward = reward
	}
}

func (a2c *A2C) Finish() {
	t := len(a2c.Actor.Trajectory) - 1
	if t >= 0 {
		a2c.Actor.Trajectory[t].Done = true
	}
}

func (a2c *A2C) Learn() {
	t := len(a2c.Actor.Trajectory) - 1

	for i := range a2c.Actor.Trajectory {
		transition := a2c.Actor.Trajectory[t-i]
		advantage := transition.Reward - a2c.Critic.Feed(transition.State)

		if !transition.Done && i != 0 {
			nextTransition := a2c.Actor.Trajectory[t-i+1]
			predNextReward := a2c.Critic.Feed(nextTransition.State)

			advantage += a2c.Actor.Gamma * predNextReward
		}

		a2c.Actor.Policy.Step(transition.State, transition.ActionGrad, advantage)
		a2c.Actor.Policy.Step(transition.State, transition.EntropyGrad, a2c.Actor.Beta)
		a2c.Critic.Step(transition.State, advantage)
	}

	a2c.Actor.Trajectory = []Transition{}
}
