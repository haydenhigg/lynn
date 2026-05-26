package lynn

import "math/rand"

type Transition struct {
	State []float64
	Error float64
}

type RL struct {
	Actor,
	Critic *Bernoulli
	Trajectory []Transition
}

func NewRL(actor, critic *Bernoulli) *RL {
	return &RL{actor, critic, []Transition{}}
}

func (rl *RL) Act(state []float64) int {
	p := rl.Actor.Prob(state)

	action := 0
	if rand.NormFloat64() < p {
		action = 1
	}

	rl.Trajectory = append(rl.Trajectory, Transition{state, float64(action) - p})

	return action
}

func (rl *RL) Reward(reward float64) {
	for _, transition := range rl.Trajectory {
		advantage := reward - rl.Critic.Feed(transition.State)

		rl.Actor.Step(advantage*transition.Error, transition.State)
		rl.Critic.Step(advantage, transition.State)
	}

	rl.Trajectory = []Transition{}
}
