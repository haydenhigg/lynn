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

	// when
	l := New(expectedD)

	// then
	if l.D != expectedD {
		t.Errorf("New(...).D != %d: %d", expectedD, l.D)
	}

	if len(l.Weights) != expectedD {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedD, len(l.Weights))
	}
}

func Test_New_negativeD(t *testing.T) {
	// when
	l := New(-2)

	// then
	expectedD := 0
	if l.D != expectedD {
		t.Errorf("New(...).D != %d: %d", expectedD, l.D)
	}

	if len(l.Weights) != expectedD {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedD, len(l.Weights))
	}
}

func Test_Linear_Feed(t *testing.T) {
	// given
	l := &Linear{
		Weights: []float64{1, -2, 3},
		Bias:    -0.5,
	}

	// when
	y := l.Feed([]float64{0.5, 0.1, 1})

	// then
	expected := 2.8
	if !almostEqual(y, expected) {
		t.Errorf("(*Linear).Feed(...) != %f: %f", expected, y)
	}
}

func Test_Linear_Step(t *testing.T) {
	// given
	l := &Linear{
		Weights: []float64{0.2, -0.8, 1.5},
		Bias:    -0.5,
	}

	// when
	l.Step([]float64{0.5, -3, 1}, 0.2)

	// then
	expectedWeights := []float64{0.3, -1.4, 1.7}
	for i, w := range l.Weights {
		if !almostEqual(w, expectedWeights[i]) {
			t.Errorf("(*Linear).Step(...).Weights[%d] != %f: %f", i, expectedWeights[i], w)
		}
	}

	expectedBias := -0.3
	if !almostEqual(l.Bias, expectedBias) {
		t.Errorf("(*Linear).Step(...).Bias != %f: %f", expectedBias, l.Bias)
	}
}

func Test_NewLinearGroup(t *testing.T) {
	// given
	expectedK := 5
	expectedD := 3

	// when
	l := NewLinearGroup(expectedK, expectedD)

	// then
	if l.K != expectedK {
		t.Errorf("NewLinearGroup(...).K != %d: %d", expectedK, l.K)
	}

	if len(l.Units) != expectedK {
		t.Errorf("len(NewLinearGroup(...).Units) != %d: %d", expectedK, len(l.Units))
	}

	for i, l := range l.Units {
		if l.D != expectedD {
			t.Errorf("len(NewLinearGroup(...).Units[%d].D) != %d: %d", i, expectedD, l.D)
		}
	}
}

func Test_NewLinearGroup_nonPositiveK(t *testing.T) {
	// when
	l := NewLinearGroup(0, 3)

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
			{Weights: []float64{-3, -2, 1}, Bias: 0.5},
			{Weights: []float64{1, 2, -3}, Bias: -0.2},
		},
	}

	// when
	l.Step([]float64{0.5, 2, 1}, []float64{1, -0.5}, 0.4)

	// then
	expectedWeights := [][]float64{
		{-2.8, -1.2, 1.4},
		{0.9, 1.6, -3.2},
	}
	for i, l := range l.Units {
		for j, w := range l.Weights {
			if !almostEqual(w, expectedWeights[i][j]) {
				t.Errorf("(*LinearGroup).Step(...).Units[%d].Weights[%d] != %f: %f", i, j, expectedWeights[i][j], w)
			}
		}
	}

	expectedBiases := []float64{0.9, -0.4}
	for i, l := range l.Units {
		if !almostEqual(l.Bias, expectedBiases[i]) {
			t.Errorf("(*LinearGroup).Step(...).Units[%d].Bias != %f: %f", i, expectedBiases[i], l.Bias)
		}
	}
}
