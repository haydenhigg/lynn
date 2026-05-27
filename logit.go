package lynn

import "math"

func Sigmoid(z float64) float64 {
	if z > 1e3 {
		return 1
	} else if z < -1e3 {
		return 0
	}

	return 1 / (1 + math.Exp(-z))
}

func Softmax(zs []float64) []float64 {
	maxZ := 0.

	for _, z := range zs {
		if z > maxZ {
			maxZ = z
		}
	}

	ps := make([]float64, len(zs))
	sum := 0.

	for i, z := range zs {
		ps[i] = math.Exp(z - maxZ)
		sum += ps[i]
	}

	for i := range ps {
		ps[i] /= sum
	}

	return ps
}
