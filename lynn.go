package lynn

import "math/rand"

type Linear struct {
	D            int
	Weights      []float64
	Bias         float64
	LearnRate    float64
	Penalty      float64
	PenaltyL1Mix float64
}

func New(d int, learnRate float64) *Linear {
	weights := make([]float64, max(d, 0))

	for i := range d {
		weights[i] = rand.NormFloat64() * learnRate
	}

	return &Linear{
		D: len(weights),
		Weights:   weights,
		Bias:      rand.NormFloat64() * learnRate,
		LearnRate: learnRate,
	}
}

func dot(as, bs []float64) float64 {
	sum := 0.

	for i, a := range as {
		sum += a * bs[i]
	}

	return sum
}

func (l *Linear) Feed(xs []float64) float64 {
	return dot(l.Weights, xs) + l.Bias
}

func sign(x float64) float64 {
	if x >= 0 {
		return 1
	} else {
		return 0
	}
}

func (l *Linear) Step(gs []float64, step float64) *Linear {
	for i, g := range gs {
		l1Penalty := l.PenaltyL1Mix * sign(l.Weights[i])
		l2Penalty := (1 - l.PenaltyL1Mix) * l.Weights[i]
		penalty := l.Penalty * (l1Penalty + l2Penalty)

		l.Weights[i] += l.LearnRate*step*g - penalty
	}

	l.Bias += l.LearnRate * step

	return l
}

func (l *Linear) Regularize(strength, l1Mix float64) *Linear {
	l.Penalty = strength
	l.PenaltyL1Mix = l1Mix
	return l
}

type LinearGroup struct {
	K     int
	Units []*Linear
}

func NewLinearGroup(k, d int, learnRate float64) *LinearGroup {
	units := make([]*Linear, max(k, 1))

	for i := range units {
		units[i] = New(d, learnRate)
	}

	return &LinearGroup{
		K:     len(units),
		Units: units,
	}
}

func (lg *LinearGroup) Feed(xs []float64) []float64 {
	ys := make([]float64, lg.K)

	for i, unit := range lg.Units {
		ys[i] = unit.Feed(xs)
	}

	return ys
}

func (lg *LinearGroup) Step(gs, unitGs []float64, step float64) *LinearGroup {
	for i, unit := range lg.Units {
		unit.Step(gs, unitGs[i]*step)
	}

	return lg
}

func (lg *LinearGroup) Regularize(strength, l1Mix float64) *LinearGroup {
	for _, unit := range lg.Units {
		unit.Regularize(strength, l1Mix)
	}

	return lg
}
