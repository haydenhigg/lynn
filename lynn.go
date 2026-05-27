package lynn

import "math/rand"

type Unit struct {
	N            int
	Weights      []float64
	Bias         float64
	LearningRate float64
}

func New(n int, learningRate float64) *Unit {
	weights := make([]float64, n)

	for i := range n {
		weights[i] = rand.NormFloat64()
	}

	bias := rand.NormFloat64()

	return &Unit{n, weights, bias, learningRate}
}

func (u *Unit) Feed(xs []float64) float64 {
	sum := 0.

	for i, w := range u.Weights {
		sum += w * xs[i]
	}

	return sum + u.Bias
}

func (u *Unit) Step(delta float64, xs []float64) {
	step := u.LearningRate * delta

	for i, x := range xs {
		u.Weights[i] += step * x
	}

	u.Bias += step
}

type Block struct {
	K int
	Units []*Unit
}

func NewBlock(k, n int, learningRate float64) *Block {
	units := make([]*Unit, max(k, 0))

	for i := range k {
		units[i] = New(n, learningRate)
	}

	return &Block{k, units}
}

func (l *Block) Feed(xs []float64) []float64 {
	ys := make([]float64, l.K)

	for i, unit := range l.Units {
		ys[i] = unit.Feed(xs)
	}

	return ys
}

func (l *Block) Step(blockDelta float64, deltas []float64, xs []float64) {
	for i, unit := range l.Units {
		unit.Step(blockDelta * deltas[i], xs)
	}
}
