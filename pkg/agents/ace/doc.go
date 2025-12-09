// Package ace implements Agentic Context Engineering (ACE) for self-improving agents.
//
// ACE enables agents to learn from their executions by maintaining a "playbook" of
// strategies, patterns, and mistakes. The playbook evolves over time as the agent
// succeeds and fails, creating a feedback loop that improves performance.
//
// # Architecture
//
// The ACE framework consists of four main components:
//
//   - Generator: Records execution trajectories during agent operation
//   - Reflector: Analyzes trajectories to extract insights and correlate feedback
//   - Curator: Manages the learnings file with deduplication and pruning
//   - Manager: Coordinates all components and handles async processing
//
// # Basic Usage
//
//	config := ace.DefaultConfig()
//	config.LearningsPath = ".learnings/agent.md"
//	config.Enabled = true
//
//	manager, err := ace.NewManager(config, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer manager.Close()
//
//	// Inject learnings into agent context
//	learningsContext := manager.GetLearningsContext()
//
//	// Record trajectory during execution
//	manager.StartTrajectory("agent-1", "code_review", "Review this PR")
//	manager.RecordStep("think", "", "Analyzing the diff [L001]...", nil, nil, nil)
//	manager.RecordStep("tool", "git_diff", "Getting changes", input, output, nil)
//	manager.EndTrajectory(ace.OutcomeSuccess)
//
// # Citation Detection
//
// When agents cite learnings in their reasoning using short codes like [L001], [M002],
// or [P003], the system tracks these citations. On success, cited learnings are marked
// as "helpful"; on failure, as "harmful". This enables credit assignment.
//
// # Learnings File Format
//
// Learnings are stored in a human-readable markdown format:
//
//	## STRATEGIES
//	[strategies-00001] helpful=5 harmful=1 :: Check nil after type assertion
//	[strategies-00002] helpful=3 harmful=0 :: Use context for cancellation
//
//	## MISTAKES
//	[mistakes-00001] helpful=2 harmful=0 :: Don't ignore Close errors
//
// # Deduplication
//
// The Curator uses tiered deduplication to avoid duplicate learnings:
//
//  1. Exact string match
//  2. Normalized match (case-insensitive, whitespace-collapsed)
//  3. Token set similarity (Jaccard index with configurable threshold)
//
// LLM-based compaction is only triggered when the file exceeds the token budget.
//
// # Configuration
//
// Key configuration options:
//
//   - LearningsPath: Where to store the learnings file
//   - AsyncReflection: Process trajectories in background (recommended for production)
//   - CurationFrequency: How many trajectories to batch before curating
//   - MinConfidence: Minimum confidence for adding new insights
//   - MaxTokens: Token budget for the learnings file (default: 80,000)
//   - PruneMinRatio: Success rate below which learnings are pruned
//   - SimilarityThreshold: Jaccard threshold for deduplication
//
// # Integration with ReActAgent
//
// To integrate ACE with ReActAgent, create a Manager and call its methods
// from the agent's execution hooks:
//
//  1. At execution start: Call GetLearningsContext() and prepend to prompt
//  2. On each step: Call RecordStep() with action details
//  3. At execution end: Call EndTrajectory() with outcome
//
// The learnings context should be prepended (not appended) to maximize LLM
// cache efficiency, as caching works on prefix matches.
//
// # Reference
//
// This implementation is based on the ACE paper:
// "Agentic Context Engineering" (arXiv:2510.04618)
package ace
