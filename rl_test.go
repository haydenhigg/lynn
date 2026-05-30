package lynn

import "testing"
import "fmt"

func Test_NewRL(t *testing.T) {
	// given
	l := NewLayer(3, 5, 1e-3)
	expectedGamma := .99
	expectedBeta := 1e-2

	// when
	rl := NewRL(l, expectedGamma, expectedBeta)

	// then
	if rl.Policy != l {
		t.Errorf("NewRL(...).Policy != %+v: %+v", l, rl.Policy)
	}

	if rl.Gamma != expectedGamma {
		t.Errorf("NewRL(...).Gamma != %f: %f", expectedGamma, rl.Gamma)
	}

	if rl.Beta != expectedBeta {
		t.Errorf("NewRL(...).Beta != %f: %f", expectedBeta, rl.Beta)
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

	fmt.Println(es)

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
