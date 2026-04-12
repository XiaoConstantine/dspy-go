package modules

type Module interface {
	Forward(inputs map[string]any) (Predict, error)
}
