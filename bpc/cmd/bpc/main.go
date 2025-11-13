package main

import (
	"bpc"
	"gpt2"
	"log"
	mbpe "mbpe-dyn"
)

func main() {
	m := model()

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	t := tokenizer()

	bpc.Run(m, t)

	if err := m.Destroy(); err != nil {
		log.Fatal(err)
	}
}

func model() *gpt2.Model {
	return gpt2.NewModel("../gpt2/models/base/model.onnx")
}

func tokenizer() *mbpe.Tokenizer {
	m := mbpe.NewMBPE()

	if err := m.Load("../gpt2/models/base/vocab.json", "../gpt2/models/base/merges.txt"); err != nil {
		panic(err)
	}

	t := mbpe.NewTokenizer(m)

	byteLevel := mbpe.NewByteLevel(false)

	t.SetPreTokenizer(byteLevel)
	t.SetDecoder(byteLevel)

	return t
}
