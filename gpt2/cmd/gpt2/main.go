package main

import (
	"fmt"
	"gpt2"
	"log"
)

func main() {
	prompt := []int64{464, 2068, 7586, 21831}

	m := gpt2.NewModel("models/base/model.onnx")

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	if out, err := m.Generate(prompt, 5, nil); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("\n%v\n", out)
	}

	if err := m.Destroy(); err != nil {
		log.Fatal(err)
	}
}
