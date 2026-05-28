package lynn

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-8
}

func Test_dot(t *testing.T) {
	y := dot([]float64{1, 2, 3}, []float64{-.1, -.2, -.3})

	expected := -1.4
	if y != expected {
		t.Errorf("dot(...) != %f: %f", expected, y)
	}
}

func Test_New(t *testing.T) {
	expectedN := 3
	expectedLearnRate := 0.1

	u := New(expectedN, expectedLearnRate)

	if len(u.Weights) != expectedN {
		t.Errorf("len(New(...).Weights) != %d: %d", expectedN, len(u.Weights))
	}
	if u.LearnRate != expectedLearnRate {
		t.Errorf("New(...).LearnRate != %f: %f", expectedLearnRate, u.LearnRate)
	}
}

func Test_Unit_Feed(t *testing.T) {
	u := &Unit{
		Weights: []float64{1, -2, 3},
		Bias:    -0.5,
	}

	y := u.Feed([]float64{0.5, 0.1, 1})

	expected := 2.8
	if !almostEqual(y, expected) {
		t.Errorf("(*Unit).Feed(...) != %f: %f", expected, y)
	}
}

func Test_Unit_Step(t *testing.T) {
	u := &Unit{
		Weights:   []float64{0.2, -0.8, 1.5},
		Bias:      -0.5,
		LearnRate: 0.1,
	}

	u.Step([]float64{0.5, -3, 1}, 2)

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
	expectedK := 5
	expectedN := 3
	expectedLearnRate := 0.1

	l := NewLayer(expectedK, expectedN, expectedLearnRate)

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

}
