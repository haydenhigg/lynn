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
	expectedD := 3
	expectedLearnRate := 0.1

	// when
	u := New(expectedD, expectedLearnRate)

	// then
	if len(u.Weights) != expectedD {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedD, len(u.Weights))
	}

	if u.LearnRate != expectedLearnRate {
		t.Errorf("New(...).LearnRate != %f: %f", expectedLearnRate, u.LearnRate)
	}
}

func Test_New_negativeD(t *testing.T) {
	// when
	u := New(-2, 0.1)

	// then
	expectedD := 0
	if len(u.Weights) != expectedD {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedD, len(u.Weights))
	}
}

func Test_Linear_Feed(t *testing.T) {
	// given
	u := &Linear{
		Weights: []float64{1, -2, 3},
		Bias:    -0.5,
	}

	// when
	y := u.Feed([]float64{0.5, 0.1, 1})

	// then
	expected := 2.8
	if !almostEqual(y, expected) {
		t.Errorf("(*Linear).Feed(...) != %f: %f", expected, y)
	}
}

func Test_Linear_Step(t *testing.T) {
	// given
	u := &Linear{
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
			t.Errorf("(*Linear).Step(...).Weights[%d] != %f: %f", i, expectedWeights[i], w)
		}
	}

	expectedBias := -0.3
	if !almostEqual(u.Bias, expectedBias) {
		t.Errorf("(*Linear).Step(...).Bias != %f: %f", expectedBias, u.Bias)
	}
}

func Test_NewLinearGroup(t *testing.T) {
	// given
	expectedK := 5
	expectedN := 3
	expectedLearnRate := 0.1

	// when
	l := NewLinearGroup(expectedK, expectedN, expectedLearnRate)

	// then
	if l.K != expectedK {
		t.Errorf("NewLinearGroup(...).K != %d: %d", expectedK, l.K)
	}

	if len(l.Units) != expectedK {
		t.Errorf("len(NewLinearGroup(...).Units) != %d: %d", expectedK, len(l.Units))
	}

	for i, u := range l.Units {
		if len(u.Weights) != expectedN {
			t.Errorf("len(NewLinearGroup(...).Units[%d].Weights) != %d: %d", i, expectedN, len(u.Weights))
		}

		if u.LearnRate != expectedLearnRate {
			t.Errorf("NewLinearGroup(...).Units[%d].LearnRate != %f: %f", i, expectedLearnRate, u.LearnRate)
		}
	}
}

func Test_NewLinearGroup_nonPositiveK(t *testing.T) {
	// when
	l := NewLinearGroup(0, 3, 0.1)

	// then
	expectedK := 1
	if l.K != expectedK {
		t.Errorf("NewLinearGroup(...).K != %d: %d", expectedK, l.K)
	}

	if len(l.Units) != expectedK {
		t.Errorf("len(NewLinearGroup(...).Units) != %d: %d", expectedK, len(l.Units))
	}
}

func Test_LinearGroup_Feed(t *testing.T) {
	// given
	l := &LinearGroup{
		K: 2,
		Units: []*Linear{
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
			t.Errorf("(*LinearGroup).Feed(...)[%d] != %f: %f", i, expected[i], y)
		}
	}
}

func Test_LinearGroup_Step(t *testing.T) {
	// given
	l := &LinearGroup{
		K: 2,
		Units: []*Linear{
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
				t.Errorf("(*LinearGroup).Step(...).Units[%d].Weights[%d] != %f: %f", i, j, expectedWeights[i][j], w)
			}
		}
	}

	expectedBiases := []float64{0.9, -0.4}
	for i, u := range l.Units {
		if !almostEqual(u.Bias, expectedBiases[i]) {
			t.Errorf("(*LinearGroup).Step(...).Units[%d].Bias != %f: %f", i, expectedBiases[i], u.Bias)
		}
	}
}
