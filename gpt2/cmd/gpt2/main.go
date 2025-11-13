package main

import (
	"fmt"
	"gpt2"
	"log"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	ort.SetSharedLibraryPath("lib/onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib")

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	defer ort.DestroyEnvironment()

	prompt := []int64{464, 2068, 7586, 21831}

	if out, err := gpt2.Generate("scripts/onnx-gpt2/model.onnx", prompt, 5, nil); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("\n%v\n", out)
	}
}
