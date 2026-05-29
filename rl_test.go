package lynn

import "testing"

func Test_NewRL(t *testing.T) {
	// given
	l := NewLayer(3, 5, 1e-3)
	expectedDiscountRate := .99

	// when
	rl := NewRL(l, expectedDiscountRate)

	// then
	if rl.Policy != l {
		t.Errorf("NewRL(...).Policy != %+v: %+v", l, rl.Policy)
	}

	if rl.DiscountRate != expectedDiscountRate {
		t.Errorf("NewRL(...).DiscountRate != %f: %f", expectedDiscountRate, rl.DiscountRate)
	}
}

func Test_probErrors(t *testing.T) {
	// given
	ps := []float64{0.1, 0.6, 0.3}
	one := 1

	// when
	es := probErrors(ps, one)

	// then
	expectedEs := []float64{-0.1, +0.4, -0.3}
	for i, e := range es {
		if !almostEqual(e, expectedEs[i]) {
			t.Errorf("probErrors(...)[%d] != %f: %f", i, expectedEs[i], e)
		}
	}
}
