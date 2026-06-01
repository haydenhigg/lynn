package lynn

import "testing"

func Test_NewRL(t *testing.T) {
	// given
	l := NewLinearGroup(3, 5, 1e-3)
	expectedDiscountRate := .99
	expectedExplorePressure := 1e-2

	// when
	rl := NewRL(l, expectedDiscountRate, expectedExplorePressure)

	// then
	if rl.Policy != l {
		t.Errorf("NewRL(...).Policy != %+v: %+v", l, rl.Policy)
	}

	if rl.DiscountRate != expectedDiscountRate {
		t.Errorf("NewRL(...).DiscountRate != %f: %f", expectedDiscountRate, rl.DiscountRate)
	}

	if rl.ExplorePressure != expectedExplorePressure {
		t.Errorf("NewRL(...).ExplorePressure != %f: %f", expectedExplorePressure, rl.ExplorePressure)
	}
}

func Test_actionErrors(t *testing.T) {
	// given
	ps := []float64{0.1, 0.6, 0.3}
	action := 1

	// when
	es := actionErrors(ps, action)

	// then
	expectedEs := []float64{-0.1, +0.4, -0.3}
	for i, e := range es {
		if !almostEqual(e, expectedEs[i]) {
			t.Errorf("actionErrors(...)[%d] != %f: %f", i, expectedEs[i], e)
		}
	}
}

func Test_entropy(t *testing.T) {
	// when
	entropy := entropy([]float64{0.1, 0.6, 0.3})

	// then
	expectedEntropy := 0.897945724
	if !almostEqual(entropy, expectedEntropy) {
		t.Errorf("entropy(...) != %f: %f", expectedEntropy, entropy)
	}
}

func Test_entropyErrors(t *testing.T) {
	// given
	ps := []float64{0.5, 0.1, 0.4}

	// when
	es := entropyErrors(ps)

	// then
	expectedEs := []float64{
		-0.113871478,
		+0.123723056,
		-0.009851577,
	}
	for i, e := range es {
		if !almostEqual(e, expectedEs[i]) {
			t.Errorf("entropyErrors(...)[%d] != %f: %f", i, expectedEs[i], e)
		}
	}
}
