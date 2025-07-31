
How Orchestrator works high level
---------------------------------

```mermaid
sequenceDiagram
    participant Client
    participant Orchestrator
    participant Analyzer
    participant TaskParser
    participant PlanCreator
    participant Processor

    Client->>Orchestrator: Process(task, context)
    activate Orchestrator

    Note over Orchestrator,Analyzer: Phase 1: Task Analysis
    Orchestrator->>Analyzer: Analyze task breakdown
    activate Analyzer
    Analyzer-->>Orchestrator: Raw analysis output (XML format)
    deactivate Analyzer

    Note over Orchestrator,TaskParser: Phase 2: Task Parsing
    Orchestrator->>TaskParser: Parse(analyzerOutput)
    activate TaskParser
    TaskParser-->>Orchestrator: Structured Task objects
    deactivate TaskParser

    Note over Orchestrator,PlanCreator: Phase 3: Plan Creation
    Orchestrator->>PlanCreator: CreatePlan(tasks)
    activate PlanCreator
    PlanCreator-->>Orchestrator: Execution phases
    deactivate PlanCreator

    Note over Orchestrator,Processor: Phase 4: Execution
    loop For each phase
        loop For each task in phase (parallel)
            Orchestrator->>Processor: Process(task, context)
            activate Processor
            Processor-->>Orchestrator: Task result
            deactivate Processor
        end
    end

    Orchestrator-->>Client: OrchestratorResult
    deactivate Orchestrator
```

Task dependency resolution
--------------------------

```mermaid

graph TD
    subgraph Task Structure
        A[Task] --> B[ID]
        A --> C[Type]
        A --> D[ProcessorType]
        A --> E[Dependencies]
        A --> F[Priority]
        A --> G[Metadata]
    end

    subgraph Plan Creation
        H[Input Tasks] --> I[Build Dependency Graph]
        I --> J[Detect Cycles]
        J --> K[Create Phases]
        K --> L[Sort by Priority]
        L --> M[Apply Max Concurrent]
    end

    subgraph Execution
        N[Phase Execution] --> O[Parallel Task Pool]
        O --> P[Process Task 1]
        O --> Q[Process Task 2]
        O --> R[Process Task N]
        P --> S[Collect Results]
        Q --> S
        R --> S
    end
```


Error handling and retry flow explained
---------------------------------------

```mermaid
stateDiagram-v2
    [*] --> TaskReceived
    TaskReceived --> Analyzing

    state Analyzing {
        [*] --> AttemptAnalysis
        AttemptAnalysis --> AnalysisSuccess
        AttemptAnalysis --> AnalysisFailure
        AnalysisFailure --> RetryAnalysis: Retry < MaxAttempts
        RetryAnalysis --> AttemptAnalysis
        AnalysisFailure --> AnalysisFailed: Retry >= MaxAttempts
    }

    state Execution {
        [*] --> ExecuteTask
        ExecuteTask --> TaskSuccess
        ExecuteTask --> TaskFailure
        TaskFailure --> RetryTask: Retry < MaxAttempts
        RetryTask --> ExecuteTask
        TaskFailure --> TaskFailed: Retry >= MaxAttempts
    }

    Analyzing --> Execution: Analysis Success
    Analyzing --> [*]: Analysis Failed
    Execution --> [*]: All Tasks Complete/Failed
```
