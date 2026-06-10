package lynn

import "math/rand"

type Linear struct {
	D       int
	Weights []float64
	Bias    float64
}

func New(d int) *Linear {
	weights := make([]float64, max(d, 0))
	return &Linear{
		D:       len(weights),
		Weights: weights,
	}
}

func (l *Linear) Randomize(mu, sigma float64) *Linear {
	for i := range l.D {
		l.Weights[i] = rand.NormFloat64()*sigma + mu
	}

	l.Bias = rand.NormFloat64()*sigma + mu

	return l
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

func (l *Linear) Step(xs []float64, scale float64) *Linear {
	for i, x := range xs {
		l.Weights[i] += scale * x
	}

	l.Bias += scale

	return l
}

type LinearGroup struct {
	K     int
	Units []*Linear
}

func NewLinearGroup(k, d int) *LinearGroup {
	units := make([]*Linear, max(k, 1))

	for i := range units {
		units[i] = New(d)
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

func (lg *LinearGroup) Step(xs, us []float64, scale float64) *LinearGroup {
	for i, unit := range lg.Units {
		unit.Step(xs, us[i]*scale)
	}

	return lg
}
