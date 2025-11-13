package llm

type Causal interface {
	Generate(prompt []int64, steps int64, logits *[][]float32) ([]int64, error)
}
