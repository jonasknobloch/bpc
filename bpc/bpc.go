package bpc

import (
	"fmt"
	"llm"
)

func Run(model llm.Causal, tokenizer llm.Tokenizer) {
	t := tokenizer.Tokenize("The quick brown")

	logits := make([][]float32, 0)

	if _, err := model.Generate(toInt64(t), 0, &logits); err != nil {
		// TODO handle
	}

	fmt.Println(logits)
}

func toInt64(s []int) []int64 {
	r := make([]int64, len(s))

	for i, v := range s {
		r[i] = int64(v)
	}

	return r
}
