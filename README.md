dspy-go
-------
[![Go Report Card](https://goreportcard.com/badge/github.com/XiaoConstantine/dspy-go)](https://goreportcard.com/report/github.com/XiaoConstantine/dspy-go)
[![codecov](https://codecov.io/gh/XiaoConstantine/dspy-go/graph/badge.svg?token=GGKRLMLXJ9)](https://codecov.io/gh/XiaoConstantine/dspy-go)


DSPy implementaion in golang


### Installation
```go
go get github.com/XiaoConstantine/dspy-go
```

### Quick Start

Here's a simple example to get you started with DSPy-Go:

```go
import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/config"
)

func main() {
    // Configure the default LLM
    err := config.ConfigureDefaultLLM("your-api-key", core.ModelAnthropicSonnet)
    if err != nil {
        log.Fatalf("Failed to configure LLM: %v", err)
    }

    // Create a signature for question answering
    signature := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.Field{Name: "answer"}}},
    )

    // Create a ChainOfThought module
    cot := modules.NewChainOfThought(signature)

    // Create a program
    program := core.NewProgram(
        map[string]core.Module{"cot": cot},
        func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
            return cot.Process(ctx, inputs)
        },
    )

    // Execute the program
    result, err := program.Execute(context.Background(), map[string]interface{}{
        "question": "What is the capital of France?",
    })
    if err != nil {
        log.Fatalf("Error executing program: %v", err)
    }

    fmt.Printf("Answer: %s\n", result["answer"])
}
```

### Core Concepts

#### Modules
Modules are the building blocks of DSPy-Go programs. They encapsulate specific functionalities and can be composed to create complex pipelines. Some key modules include:

* Predict: Basic prediction module
* ChainOfThought: Implements chain-of-thought reasoning
* ReAct: Implements the ReAct (Reasoning and Acting) paradigm


#### Signatures
Signatures define the input and output fields for modules. They help in creating type-safe and well-defined interfaces for your AI components.


#### Program
Programs in DSPy-Go represent complete AI workflows. They combine multiple modules and define how data flows between them.


#### Optimizers
Optimizers help improve the performance of your DSPy-Go programs by automatically tuning prompts and module parameters. The BootstrapFewShot optimizer is currently implemented.


### License
DSPy-Go is released under the MIT License. See the LICENSE file for details.
