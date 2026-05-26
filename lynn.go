package lynn

import (
	"math"
	"math/rand"
)

type Bernoulli struct {
	N int
	Weights []float64
	Bias float64
	LearningRate float64
}

func NewBernoulli(n int, learningRate float64) *Bernoulli {
	weights := make([]float64, n)
	for i := range n {
		weights[i] = rand.NormFloat64()
	}

	bias := rand.NormFloat64()

	return &Bernoulli{n, weights, bias, learningRate}
}

func (b *Bernoulli) Feed(xs []float64) float64 {
	dot := 0.
	for i, x := range xs {
		dot += x * b.Weights[i]
	}

	return dot + b.Bias
}

func (b *Bernoulli) Prob(xs []float64) float64 {
	z := b.Feed(xs)
	return 1 / (1 + math.Exp(-z))
}

func (b *Bernoulli) Step(d float64, xs []float64) {
	step := b.LearningRate * d

	for i, x := range xs {
		b.Weights[i] += step * x
	}

	b.Bias += step
}

type Multinomial struct {
	K int
	Models []*Bernoulli
}

func NewMultinomial(k, n int, learningRate float64) *Multinomial {
	models := make([]*Bernoulli, k)
	for i := range k {
		models[i] = NewBernoulli(n, learningRate)
	}

	return &Multinomial{k, models}
}

func (m *Multinomial) Feed(xs []float64) []float64 {
	outputs := make([]float64, m.K)
	for i, model := range m.Models {
		outputs[i] = model.Feed(xs)
	}

	return outputs
}

// func (m *Multinomial) Prob(xs []float64) []float64 {
// 	ps := make([]float64, m.K)
// 	sum := 0.
// 	for i, y := range m.Feed(xs) {
// 		ps[i] = math.Exp(y)
// 		sum += ps[i]
// 	}

// 	for i := range m.K {
// 		ps[i] /= sum
// 	}

// 	return ps
// }

// func (m *Multinomial) Step(ds []float64, xs []float64) {
// 	for j, model := range m.Models {
// 		model.Step(ds[j], xs)
// 	}
// }
