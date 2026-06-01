package lynn

import (
	"math"
	"math/rand"
)

type Transition struct {
	State           []float64
	ActionGradient  []float64
	EntropyGradient []float64
	Reward          float64
	Done            bool
}

type RL struct {
	Policy          *LinearGroup
	DiscountRate    float64
	ExplorePressure float64
	Trajectory      []Transition
}

func NewRL(policy *LinearGroup, discountRate float64) *RL {
	return &RL{
		Policy:       policy,
		DiscountRate: discountRate,
		Trajectory:   []Transition{},
	}
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
	normalizer := math.Log(float64(len(ps)))

	errors := make([]float64, len(ps))

	for i, p := range ps {
		errors[i] = -p * (math.Log(p) + h) / normalizer
	}

	return errors
}

func (rl *RL) Act(state []float64) int {
	ps := Softmax(rl.Policy.Feed(state))
	action := sampleAction(ps)

	rl.Trajectory = append(rl.Trajectory, Transition{
		State:           state,
		ActionGradient:  actionErrors(ps, action),
		EntropyGradient: entropyErrors(ps),
	})

	return action
}

func (rl *RL) applyReward(transition Transition, reward float64) {
	rl.Policy.Step(transition.State, transition.ActionGradient, reward)

	if rl.ExplorePressure > 0 {
		rl.Policy.Step(transition.State, transition.EntropyGradient, rl.ExplorePressure)
	}
}

func (rl *RL) Reward(reward float64) *RL {
	t := len(rl.Trajectory) - 1
	discountFactor := 1.

	for i := range rl.Trajectory {
		rl.applyReward(rl.Trajectory[t-i], reward*discountFactor)
		discountFactor *= rl.DiscountRate
	}

	rl.Trajectory = []Transition{}

	return rl
}

func (rl *RL) Regularize(strength float64) *RL {
	rl.ExplorePressure = strength
	return rl
}

type A2C struct {
	Actor  *RL
	Critic *Linear
}

func NewA2C(actor *RL, critic *Linear) *A2C {
	return &A2C{actor, critic}
}

func (a2c *A2C) Act(state []float64) int {
	return a2c.Actor.Act(state)
}

func (a2c *A2C) Reward(reward float64) *A2C {
	t := len(a2c.Actor.Trajectory) - 1
	if t >= 0 {
		a2c.Actor.Trajectory[t].Reward = reward
	}

	return a2c
}

func (a2c *A2C) Finish() *A2C {
	t := len(a2c.Actor.Trajectory) - 1
	if t >= 0 {
		a2c.Actor.Trajectory[t].Done = true
	}

	return a2c
}

func (a2c *A2C) Learn() *A2C {
	t := len(a2c.Actor.Trajectory) - 1

	for i := range a2c.Actor.Trajectory {
		transition := a2c.Actor.Trajectory[t-i]
		advantage := transition.Reward - a2c.Critic.Feed(transition.State)

		if !transition.Done && i != 0 {
			nextTransition := a2c.Actor.Trajectory[t-i+1]
			predNextReward := a2c.Critic.Feed(nextTransition.State)

			advantage += a2c.Actor.DiscountRate * predNextReward
		}

		a2c.Actor.applyReward(transition, advantage)
		a2c.Critic.Step(transition.State, advantage)
	}

	a2c.Actor.Trajectory = []Transition{}

	return a2c
}
