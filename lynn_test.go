package lynn

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-8
}

func Test_dot(t *testing.T) {
	// when
	y := dot([]float64{1, 2, 3}, []float64{-.1, -.2, -.3})

	// then
	expected := -1.4
	if y != expected {
		t.Errorf("dot(...) != %f: %f", expected, y)
	}
}

func Test_New(t *testing.T) {
	// given
	expectedN := 3
	expectedLearnRate := 0.1

	// when
	u := New(expectedN, expectedLearnRate)

	// then
	if len(u.Weights) != expectedN {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedN, len(u.Weights))
	}

	if u.LearnRate != expectedLearnRate {
		t.Errorf("New(...).LearnRate != %f: %f", expectedLearnRate, u.LearnRate)
	}
}

func Test_Unit_Feed(t *testing.T) {
	// given
	u := &Unit{
		Weights: []float64{1, -2, 3},
		Bias:    -0.5,
	}

	// when
	y := u.Feed([]float64{0.5, 0.1, 1})

	// then
	expected := 2.8
	if !almostEqual(y, expected) {
		t.Errorf("(*Unit).Feed(...) != %f: %f", expected, y)
	}
}

func Test_Unit_Step(t *testing.T) {
	// given
	u := &Unit{
		Weights:   []float64{0.2, -0.8, 1.5},
		Bias:      -0.5,
		LearnRate: 0.1,
	}

	// when
	u.Step([]float64{0.5, -3, 1}, 2)

	// then
	expectedWeights := []float64{0.3, -1.4, 1.7}
	for i, w := range u.Weights {
		if !almostEqual(w, expectedWeights[i]) {
			t.Errorf("(*Unit).Step(...).Weights[%d] != %f: %f", i, expectedWeights[i], w)
		}
	}

	expectedBias := -0.3
	if !almostEqual(u.Bias, expectedBias) {
		t.Errorf("(*Unit).Step(...).Bias != %f: %f", expectedBias, u.Bias)
	}
}

func Test_NewLayer(t *testing.T) {
	// given
	expectedK := 5
	expectedN := 3
	expectedLearnRate := 0.1

	// when
	l := NewLayer(expectedK, expectedN, expectedLearnRate)

	// then
	if l.K != expectedK {
		t.Errorf("NewLayer(...).K != %d: %d", expectedK, l.K)
	}

	for i, u := range l.Units {
		if len(u.Weights) != expectedN {
			t.Errorf("len(NewLayer(...).Units[%d].Weights) != %d: %d", i, expectedN, len(u.Weights))
		}

		if u.LearnRate != expectedLearnRate {
			t.Errorf("NewLayer(...).Units[%d].LearnRate != %f: %f", i, expectedLearnRate, u.LearnRate)
		}
	}
}

func Test_Layer_Feed(t *testing.T) {
	// given
	l := &Layer{
		K: 2,
		Units: []*Unit{
			{Weights: []float64{1, -2, 3}, Bias: -0.5},
			{Weights: []float64{-1, 0.5, 2}, Bias: -1.5},
		},
	}

	// when
	ys := l.Feed([]float64{0.5, -3, 0.2})

	// then
	expected := []float64{6.6, -3.1}
	for i, y := range ys {
		if !almostEqual(y, expected[i]) {
			t.Errorf("(*Layer).Feed(...)[%d] != %f: %f", i, expected[i], y)
		}
	}
}

func Test_Layer_Step(t *testing.T) {
	// given
	l := &Layer{
		K: 2,
		Units: []*Unit{
			{Weights: []float64{-3, -2, 1}, Bias: 0.5, LearnRate: 0.1},
			{Weights: []float64{1, 2, -3}, Bias: -0.2, LearnRate: 0.2},
		},
	}

	// when
	l.Step([]float64{0.5, 2, 1}, []float64{2, -0.5}, 2)

	// then
	expectedWeights := [][]float64{
		{-2.8, -1.2, 1.4},
		{0.9, 1.6, -3.2},
	}
	for i, u := range l.Units {
		for j, w := range u.Weights {
			if !almostEqual(w, expectedWeights[i][j]) {
				t.Errorf("(*Layer).Step(...).Units[%d].Weights[%d] != %f: %f", i, j, expectedWeights[i][j], w)
			}
		}
	}

	expectedBiases := []float64{0.9, -0.4}
	for i, u := range l.Units {
		if !almostEqual(u.Bias, expectedBiases[i]) {
			t.Errorf("(*Layer).Step(...).Units[%d].Bias != %f: %f", i, expectedBiases[i], u.Bias)
		}
	}
}
