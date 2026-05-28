package lynn

import "math/rand"

type Unit struct {
	Weights   []float64
	Bias      float64
	LearnRate float64
}

func New(n int, learnRate float64) *Unit {
	weights := make([]float64, max(n, 0))

	for i := range n {
		weights[i] = rand.NormFloat64()
	}

	bias := rand.NormFloat64()

	return &Unit{weights, bias, learnRate}
}

func dot(as, bs []float64) float64 {
	sum := 0.

	for i, a := range as {
		sum += a * bs[i]
	}

	return sum
}

func (u *Unit) Feed(xs []float64) float64 {
	return dot(u.Weights, xs) + u.Bias
}

func (u *Unit) Step(gs []float64, step float64) {
	alpha := u.LearnRate * step

	for i, g := range gs {
		u.Weights[i] += alpha * g
	}

	u.Bias += alpha
}

type Layer struct {
	K     int
	Units []*Unit
}

func NewLayer(k, n int, learnRate float64) *Layer {
	units := make([]*Unit, max(k, 1))

	for i := range units {
		units[i] = New(n, learnRate)
	}

	return &Layer{len(units), units}
}

func (l *Layer) Feed(xs []float64) []float64 {
	ys := make([]float64, l.K)

	for i, unit := range l.Units {
		ys[i] = unit.Feed(xs)
	}

	return ys
}

func (l *Layer) Step(gs []float64, ds []float64, alpha float64) {
	for i, unit := range l.Units {
		unit.Step(gs, ds[i]*alpha)
	}
}
