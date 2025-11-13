package gpt2

import (
	"errors"
	"fmt"
	_ "llm"
	"log"
	"math"
	"os"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	vocabSize = 50257
	nLayers   = 12
	nHeads    = 12
	headDim   = 64
)

type Model struct {
	name string
}

func NewModel(name string) *Model {
	return &Model{
		name: name,
	}
}

func (m *Model) SharedLibraryPath() string {
	p, ok := os.LookupEnv("ONNXRUNTIME_SHARED_LIBRARY_PATH")

	if !ok {
		// TODO embed runtime binaries
	}

	return p
}

func (m *Model) Init() error {
	ort.SetSharedLibraryPath(m.SharedLibraryPath())

	return ort.InitializeEnvironment()
}

func (m *Model) Destroy() error {
	return ort.DestroyEnvironment()
}

func (m *Model) Generate(prompt []int64, steps int64, logits *[][]float32) ([]int64, error) {
	if len(prompt) == 0 {
		return nil, errors.New("empty prompt")
	}

	context := int64(len(prompt))

	cacheNames, cacheValues := emptyCache()

	token := prompt[0]

	out := make([]int64, 0, steps+1)

	for step := range context + steps {
		_, _, outputs, err := forward(m.name, token, step, cacheNames, cacheValues)

		if err != nil {
			return nil, err
		}

		l := outputs[0].(*ort.Tensor[float32]).GetData()

		if logits != nil {
			*logits = append(*logits, l)
		}

		idx, _ := topK(softmax(l), 5)

		// fmt.Printf("\n%d\n\n", token)

		// for i, t := range idx {
		// 	fmt.Printf("%.4f %.4f [%d]\n", l[t], p[i], t)
		// }

		if step < context-1 {
			token = prompt[step+1]
		} else {
			token = int64(idx[0]) // choose best token
			out = append(out, token)
		}

		cacheValues = outputs[1:]
	}

	return out[:steps], nil
}

func forward(model string, token int64, position int64, cacheNames []string, cacheValues []ort.Value) (*ort.Tensor[float32], []string, []ort.Value, error) {
	inputNames, inputs, _ := initInputs(token, position)
	outputNames, outputs, logits, _ := initOutputs(position)

	inputNames = append(inputNames, cacheNames...)
	inputs = append(inputs, cacheValues...)

	session, err := ort.NewAdvancedSession(
		model,
		inputNames,
		outputNames,
		inputs,
		outputs,
		nil,
	)

	if err != nil {
		log.Fatal(err)
	}

	defer session.Destroy()

	if err := session.Run(); err != nil {
		log.Fatal(err)
	}

	return logits, outputNames, outputs, nil
}

func emptyCache() ([]string, []ort.Value) {
	names := make([]string, 0, 2*nLayers)
	values := make([]ort.Value, 0, 2*nLayers)
	shape := []int64{1, int64(nHeads), 0, int64(headDim)}

	for i := range nLayers {
		kName := fmt.Sprintf("past_key_values.%d.key", i)
		vName := fmt.Sprintf("past_key_values.%d.value", i)

		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		names = append(names, kName, vName)
		values = append(values, ort.Value(kTensor), ort.Value(vTensor))
	}

	return names, values
}

func initInputs(token, position int64) ([]string, []ort.Value, error) {
	inputNames := []string{"input_ids", "position_ids", "attention_mask"}

	var tokens *ort.Tensor[int64]
	var positions *ort.Tensor[int64]
	var attentionMask *ort.Tensor[int64]

	if t, err := ort.NewTensor[int64]([]int64{1, 1}, []int64{token}); err != nil {
		return nil, nil, err
	} else {
		tokens = t
	}

	if p, err := ort.NewTensor[int64]([]int64{1, 1}, []int64{position}); err != nil {
		return nil, nil, err
	} else {
		positions = p
	}

	maskData := make([]int64, position+1)
	maskShape := []int64{1, position + 1}

	for i := range maskData {
		maskData[i] = 1
	}

	if m, err := ort.NewTensor[int64](maskShape, maskData); err != nil {
		return nil, nil, err
	} else {
		attentionMask = m
	}

	inputs := []ort.Value{ort.Value(tokens), ort.Value(positions), ort.Value(attentionMask)}

	return inputNames, inputs, nil
}

func initOutputs(position int64) ([]string, []ort.Value, *ort.Tensor[float32], error) {
	outputNames := make([]string, 0, 1+2*nLayers)
	outputValues := make([]ort.Value, 0, 1+2*nLayers)

	logits, err := ort.NewEmptyTensor[float32]([]int64{1, 1, int64(vocabSize)})

	if err != nil {
		return nil, nil, nil, err
	}

	outputNames = append(outputNames, "logits")
	outputValues = append(outputValues, ort.Value(logits))

	shape := []int64{1, int64(nHeads), position + 1, int64(headDim)}

	for i := range nLayers {
		kName := fmt.Sprintf("present.%d.key", i)
		vName := fmt.Sprintf("present.%d.value", i)

		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		outputNames = append(outputNames, kName, vName)
		outputValues = append(outputValues, ort.Value(kTensor), ort.Value(vTensor))
	}

	return outputNames, outputValues, logits, nil
}

func softmax(logits []float32) []float32 {
	m := logits[0]

	for _, v := range logits {
		if v > m {
			m = v
		}
	}

	s := float32(0.0)
	r := make([]float32, len(logits))

	for i, v := range logits {
		e := float32(math.Exp(float64(v - m)))

		r[i] = e
		s += e
	}

	for i := range r {
		r[i] /= s
	}

	return r
}

func topK(p []float32, k int) ([]int, []float32) {
	n := len(p)

	if k > n {
		k = n
	}

	idx := make([]int, n)

	for i := range idx {
		idx[i] = i
	}

	sort.Slice(idx, func(i, j int) bool {
		return p[idx[i]] > p[idx[j]]
	})

	topIdx := idx[:k]
	topP := make([]float32, k)

	for i := 0; i < k; i++ {
		topP[i] = p[topIdx[i]]
	}

	return topIdx, topP
}
