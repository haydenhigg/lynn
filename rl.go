package lynn

import (
	"math"
	"math/rand"
)

type Transition struct {
	State  []float64
	Errors []float64
}

type RL struct {
	Policy       *Layer
	Trajectory   []Transition
	DiscountRate float64
}

func NewRL(policy *Layer, discountRate float64) *RL {
	return &RL{policy, []Transition{}, discountRate}
}

func probSample(ps []float64) int {
	v := rand.Float64()

	for i, p := range ps {
		if v -= p; v < 0 {
			return i
		}
	}

	return -1
}

func probErrors(ps []float64, one int) []float64 {
	es := make([]float64, len(ps))

	for i, p := range ps {
		if one == i {
			es[i] = 1 - p
		} else {
			es[i] = 0 - p
		}
	}

	return es
}

func (rl *RL) Act(state []float64) int {
	ps := Softmax(rl.Policy.Feed(state))
	action := probSample(ps)

	rl.Trajectory = append(rl.Trajectory, Transition{
		state,
		probErrors(ps, action),
	})

	return action
}

func (rl *RL) Reward(reward float64) {
	t := len(rl.Trajectory) - 1

	for i, transition := range rl.Trajectory {
		discount := math.Pow(rl.DiscountRate, float64(t-i))
		rl.Policy.Step(transition.State, transition.Errors, reward*discount)
	}

	rl.Trajectory = []Transition{}
}

type A2C struct {
	Actor  *RL
	Critic *Unit
}

func NewA2C(policy *Layer, critic *Unit, discountRate float64) *A2C {
	return &A2C{NewRL(policy, discountRate), critic}
}

func (a2c *A2C) Act(state []float64) int {
	return a2c.Actor.Act(state)
}

func (a2c *A2C) Reward(reward float64) {
	t := len(a2c.Actor.Trajectory) - 1

	for i, transition := range a2c.Actor.Trajectory {
		discount := math.Pow(a2c.Actor.DiscountRate, float64(t-i))
		advantage := reward*discount - a2c.Critic.Feed(transition.State)

		a2c.Actor.Policy.Step(transition.State, transition.Errors, advantage)
		a2c.Critic.Step(transition.State, advantage)
	}

	a2c.Actor.Trajectory = []Transition{}
}
