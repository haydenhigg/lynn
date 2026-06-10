package lynn

import "testing"

func Test_GradientSolver_Ascend(t *testing.T) {
	// given
	l := New(3)
	gs := NewGradientSolver(1e-1)

	// when
	gs.Ascend(l, []float64{1, 2, 3}, 0.5)

	// then
	expectedWeights := []float64{0.05, 0.1, 0.15}
	for i, w := range l.Weights {
		if !almostEqual(w, expectedWeights[i]) {
			t.Errorf("(*Linear).Weights[%d] != %f: %f", i, expectedWeights[i], w)
		}
	}

	expectedBias := 0.05
	if !almostEqual(l.Bias, expectedBias) {
		t.Errorf("(*Linear).Bias != %f: %f", expectedBias, l.Bias)
	}
}

func Test_GradientSolver_Ascend_regularized(t *testing.T) {
	// given
	l := New(3)
	gs := NewGradientSolver(1e-1)

	gs.L1Penalty = 0.4
	gs.L2Penalty = 0.6

	// when
	gs.Ascend(l, []float64{1, 2, 3}, 0.5)
	gs.Ascend(l, []float64{-3, 2, -1}, 0.8)

	// then
	expectedWeights := []float64{-0.233, 0.214, 0.021}
	for i, w := range l.Weights {
		if !almostEqual(w, expectedWeights[i]) {
			t.Errorf("(*Linear).Weights[%d] != %f: %f", i, expectedWeights[i], w)
		}
	}

	expectedBias := 0.13
	if !almostEqual(l.Bias, expectedBias) {
		t.Errorf("(*Linear).Bias != %f: %f", expectedBias, l.Bias)
	}
}
