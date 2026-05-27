package lynn

import "math/rand"

type Transition struct {
	State  []float64
	Errors []float64
}

type RL struct {
	Actor *Block
	Critic *Unit
	Trajectory []Transition
}

func NewRL(k, n int, actorLearningRate, criticLearningRate float64) *RL {
	return &RL{
		NewBlock(k, n, actorLearningRate),
		New(n, criticLearningRate),
		[]Transition{},
	}
}

// ps must sum to 1
func sample(ps []float64) int {
	v := rand.Float64()

	for i, p := range ps {
		v -= p

		if v < 0 {
			return i
		}
	}

	return -1
}

func (rl *RL) Act(state []float64) int {
	ps := Softmax(rl.Actor.Feed(state))

	action := sample(ps)
	errors := make([]float64, len(ps))

	for i, p := range ps {
		if action == i {
			errors[i] = 1 - p
		} else {
			errors[i] = 0 - p
		}
	}

	rl.Trajectory = append(rl.Trajectory, Transition{state, errors})

	return action
}

func (rl *RL) Reward(reward float64) {
	for _, transition := range rl.Trajectory {
		advantage := reward - rl.Critic.Feed(transition.State)

		rl.Actor.Step(advantage, transition.Errors, transition.State)
		rl.Critic.Step(advantage, transition.State)
	}

	rl.Trajectory = []Transition{}
}
