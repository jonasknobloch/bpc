package llm

type Tokenizer interface {
	Tokenize(s string) []int
}
