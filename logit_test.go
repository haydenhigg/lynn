package lynn

import "testing"

func Test_Sigmoid(t *testing.T) {
	// when
	y := Sigmoid(1.82)

	// then
	expected := 0.860566127
	if !almostEqual(y, expected) {
		t.Errorf("Sigmoid(...) != %f: %f", expected, y)
	}
}

func Test_Softmax(t *testing.T) {
	// when
	ys := Softmax([]float64{-2.8, 3.9, 4.1})

	// then
	expected := []float64{
		0.000553807,
		0.449916697,
		0.549529494,
	}
	for i, y := range ys {
		if !almostEqual(y, expected[i]) {
			t.Errorf("Softmax(...)[%d] != %f: %f", i, expected[i], y)
		}
	}
}
