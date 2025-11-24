package gpt2

import (
	"errors"
	"fmt"
	_ "llm"
	"math"
	"os"
	"slices"
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
	name        string
	deviceID    string
	session     *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
}

func NewModel(name, deviceID string) *Model {
	return &Model{
		name:     name,
		deviceID: deviceID,
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

	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}

	inputNames := make([]string, 0, 3+2*nLayers)
	outputNames := make([]string, 0, 1+2*nLayers)

	inputNames = append(inputNames, "input_ids", "position_ids", "attention_mask")
	outputNames = append(outputNames, "logits")

	for i := range nLayers {
		inputNames = append(inputNames, fmt.Sprintf("past_key_values.%d.key", i), fmt.Sprintf("past_key_values.%d.value", i))
		outputNames = append(outputNames, fmt.Sprintf("present.%d.key", i), fmt.Sprintf("present.%d.value", i))
	}

	m.inputNames = inputNames
	m.outputNames = outputNames

	var options *ort.SessionOptions

	if m.deviceID != "" {
		if opts, err := SessionsOptionsWithCUDADeviceID(m.deviceID); err != nil {
			return err
		} else {
			options = opts

			defer options.Destroy()
		}
	}

	if s, err := ort.NewDynamicAdvancedSession(m.name, m.inputNames, m.outputNames, options); err != nil {
		return err
	} else {
		m.session = s
	}

	return nil
}

func (m *Model) Destroy() error {
	return ort.DestroyEnvironment()
}

func (m *Model) Generate(prompt []int64, steps int64, logits *[][]float32) ([]int64, error) {
	if len(prompt) == 0 {
		return nil, errors.New("empty prompt")
	}

	context := int64(len(prompt))

	cacheValues := emptyCache()

	defer func() {
		destroyValues(cacheValues)
	}()

	token := prompt[0]

	out := make([]int64, 0, steps+1)

	for step := range context + steps {
		outputs, err := m.forward(token, step, cacheValues)

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

		_ = outputs[0].Destroy()

		destroyValues(cacheValues)

		cacheValues = outputs[1:]
	}

	return out[:steps], nil
}

func (m *Model) forward(token int64, position int64, cacheValues []ort.Value) ([]ort.Value, error) {
	var binding *ort.IoBinding

	if b, err := m.session.CreateIoBinding(); err != nil {
		return nil, err
	} else {
		binding = b

		defer binding.Destroy()
	}

	var inputs []ort.Value
	var outputs []ort.Value

	if in, err := initInputs(token, position); err != nil {
		return nil, err
	} else {
		inputs = in
	}

	defer destroyValues(inputs)

	if out, err := initOutputs(position); err != nil {
		return nil, err
	} else {
		outputs = out
	}

	inputs = append(inputs, cacheValues...)

	if len(inputs) != len(m.inputNames) {
		panic("unexpected input length")
	}

	for i, name := range m.inputNames {
		if err := binding.BindInput(name, inputs[i]); err != nil {
			return nil, err
		}
	}

	var ok bool

	defer func() {
		if !ok {
			destroyValues(outputs)
		}
	}()

	if len(outputs) != len(m.outputNames) {
		panic("unexpected output length")
	}

	for i, name := range m.outputNames {
		if err := binding.BindOutput(name, outputs[i]); err != nil {
			return nil, err
		}
	}

	if err := m.session.RunWithBinding(binding); err != nil {
		return nil, err
	}

	ok = true

	return outputs, nil
}

func destroyValues(values []ort.Value) {
	for _, v := range values {
		_ = v.Destroy()
	}
}

func emptyCache() []ort.Value {
	values := make([]ort.Value, 0, 2*nLayers)
	shape := []int64{1, int64(nHeads), 0, int64(headDim)}

	for range nLayers {
		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		values = append(values, ort.Value(kTensor), ort.Value(vTensor))
	}

	return values
}

func initInputs(token, position int64) ([]ort.Value, error) {
	var tokens *ort.Tensor[int64]
	var positions *ort.Tensor[int64]
	var attentionMask *ort.Tensor[int64]

	if t, err := ort.NewTensor[int64]([]int64{1, 1}, []int64{token}); err != nil {
		return nil, err
	} else {
		tokens = t
	}

	if p, err := ort.NewTensor[int64]([]int64{1, 1}, []int64{position}); err != nil {
		return nil, err
	} else {
		positions = p
	}

	maskData := make([]int64, position+1)
	maskShape := []int64{1, position + 1}

	for i := range maskData {
		maskData[i] = 1
	}

	if m, err := ort.NewTensor[int64](maskShape, maskData); err != nil {
		return nil, err
	} else {
		attentionMask = m
	}

	inputs := []ort.Value{ort.Value(tokens), ort.Value(positions), ort.Value(attentionMask)}

	return inputs, nil
}

func initOutputs(position int64) ([]ort.Value, error) {
	outputs := make([]ort.Value, 0, 1+2*nLayers)

	logits, err := ort.NewEmptyTensor[float32]([]int64{1, 1, int64(vocabSize)})

	if err != nil {
		return nil, err
	}

	outputs = append(outputs, ort.Value(logits))

	shape := []int64{1, int64(nHeads), position + 1, int64(headDim)}

	for range nLayers {
		kTensor, _ := ort.NewEmptyTensor[float32](shape)
		vTensor, _ := ort.NewEmptyTensor[float32](shape)

		outputs = append(outputs, ort.Value(kTensor), ort.Value(vTensor))
	}

	return outputs, nil
}

func softmax(logits []float32) []float32 {
	m := slices.Max(logits)

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
