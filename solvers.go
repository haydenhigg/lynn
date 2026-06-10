package lynn

type GradientSolver struct {
	LearnRate float64
	L1Penalty float64
	L2Penalty float64
}

func NewGradientSolver(learnRate float64) *GradientSolver {
	return &GradientSolver{LearnRate: learnRate}
}

func (gs *GradientSolver) Lasso(strength float64) *GradientSolver {
	gs.L1Penalty = strength
	return gs
}

func (gs *GradientSolver) Ridge(strength float64) *GradientSolver {
	gs.L2Penalty = strength
	return gs
}

func (gs *GradientSolver) ElasticNet(strength, l1Mix float64) *GradientSolver {
	gs.L1Penalty = strength * l1Mix
	gs.L2Penalty = strength * (1 - l1Mix)
	return gs
}

func sign(x float64) float64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	} else {
		return 0
	}
}

func (gs *GradientSolver) Ascend(l *Linear, xs []float64, scale float64) *GradientSolver {
	if gs.L1Penalty != 0 || gs.L2Penalty != 0 {
		for i, w := range l.Weights {
			l.Weights[i] -= gs.L1Penalty*sign(w) + gs.L2Penalty*w
		}
	}

	l.Step(xs, scale*gs.LearnRate)

	return gs
}

func (gs *GradientSolver) Descend(l *Linear, xs []float64, scale float64) *GradientSolver {
	return gs.Ascend(l, xs, -scale)
}
