package modules

type Module interface {
	Forward(inputs map[string]interface{}) (Predict, error)
}
