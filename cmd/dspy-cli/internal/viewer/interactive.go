package viewer

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"
)

// RunInteractive starts the interactive viewer mode.
func RunInteractive(data *LogData, cfg Config) {
	switch data.Format {
	case FormatRLM:
		runRLMInteractive(data, cfg)
	case FormatDSPy:
		runDSPyInteractive(data, cfg)
	default:
		fmt.Println("Unknown log format")
	}
}

// runRLMInteractive handles interactive mode for RLM format logs.
func runRLMInteractive(data *LogData, cfg Config) {
	if len(data.Iterations) == 0 {
		fmt.Println("No iterations to display")
		return
	}

	// Save terminal state
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		fmt.Printf("Error entering raw mode: %v\n", err)
		ViewLog(data, cfg)
		return
	}
	defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }()

	currentIdx := 0
	expanded := !cfg.Compact
	searchQuery := cfg.Search
	searchResults := []int{}
	searchIdx := 0

	if searchQuery != "" {
		searchResults = FindSearchMatches(data.Iterations, searchQuery)
		if len(searchResults) > 0 {
			currentIdx = searchResults[0]
		}
	}

	ClearScreen()
	printInteractiveIteration(data, currentIdx, expanded, searchQuery, len(searchResults), searchIdx)

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			break
		}

		switch n {
		case 1:
			switch buf[0] {
			case 'q', 3: // q or Ctrl+C
				ClearScreen()
				return
			case 'j':
				if currentIdx < len(data.Iterations)-1 {
					currentIdx++
				}
			case 'k':
				if currentIdx > 0 {
					currentIdx--
				}
			case 'g':
				currentIdx = 0
			case 'G':
				currentIdx = len(data.Iterations) - 1
			case 'e':
				expanded = !expanded
			case 'n':
				if len(searchResults) > 0 {
					searchIdx = (searchIdx + 1) % len(searchResults)
					currentIdx = searchResults[searchIdx]
				}
			case 'N':
				if len(searchResults) > 0 {
					searchIdx = (searchIdx - 1 + len(searchResults)) % len(searchResults)
					currentIdx = searchResults[searchIdx]
				}
			case '/':
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				fmt.Print("\r\033[K/")
				reader := bufio.NewReader(os.Stdin)
				query, _ := reader.ReadString('\n')
				searchQuery = strings.TrimSpace(query)
				searchResults = FindSearchMatches(data.Iterations, searchQuery)
				searchIdx = 0
				if len(searchResults) > 0 {
					currentIdx = searchResults[0]
				}
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			case 's':
				// Show stats
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				ClearScreen()
				PrintDetailedStats(data)
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			case '?':
				// Show help
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				ClearScreen()
				printInteractiveHelp()
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			}
		case 3:
			// Arrow keys
			if buf[0] == 27 && buf[1] == 91 {
				switch buf[2] {
				case 65: // Up
					if currentIdx > 0 {
						currentIdx--
					}
				case 66: // Down
					if currentIdx < len(data.Iterations)-1 {
						currentIdx++
					}
				}
			}
		}

		ClearScreen()
		printInteractiveIteration(data, currentIdx, expanded, searchQuery, len(searchResults), searchIdx)
	}
}

func printInteractiveIteration(data *LogData, idx int, expanded bool, searchQuery string, matchCount, matchIdx int) {
	iter := data.Iterations[idx]

	// Status bar
	fmt.Printf("%s%s Iteration %d/%d %s", BgBlue, White, idx+1, len(data.Iterations), Reset)
	if matchCount > 0 {
		fmt.Printf(" %s[Match %d/%d]%s", Dim, matchIdx+1, matchCount, Reset)
	}
	if iter.FinalAnswer != nil {
		fmt.Printf(" %s[FINAL]%s", BoldGreen, Reset)
	}
	fmt.Printf(" %s[e]xpand [/]search [s]tats [?]help [q]uit%s\n", Dim, Reset)
	fmt.Println(strings.Repeat("─", 60))

	// Print the iteration
	if expanded {
		PrintIteration(iter, false, searchQuery)
	} else {
		PrintIteration(iter, true, searchQuery)
	}

	// Navigation hints
	fmt.Printf("\n%s← k/↑  j/↓ →  g=first G=last  n/N=search%s\n", Dim, Reset)
}

func printInteractiveHelp() {
	fmt.Printf(`
%s%s Interactive Mode Help %s

%sNavigation:%s
  j, ↓       Next iteration
  k, ↑       Previous iteration
  g          Go to first iteration
  G          Go to last iteration

%sSearch:%s
  /          Enter search query
  n          Next search result
  N          Previous search result

%sDisplay:%s
  e          Toggle expand/compact mode
  s          Show detailed statistics

%sOther:%s
  ?          Show this help
  q, Ctrl+C  Quit

`, BoldCyan, "═══", Reset, Bold, Reset, Bold, Reset, Bold, Reset, Bold, Reset)
}

// runDSPyInteractive handles interactive mode for native DSPy format logs.
func runDSPyInteractive(data *LogData, cfg Config) {
	// Count navigable items
	totalItems := len(data.LLMCalls) + len(data.Modules) + len(data.CodeExecs) + len(data.ToolCalls)
	if totalItems == 0 {
		fmt.Println("No events to display")
		return
	}

	// Save terminal state
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		fmt.Printf("Error entering raw mode: %v\n", err)
		ViewLog(data, cfg)
		return
	}
	defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }()

	// Build a list of viewable items
	type viewItem struct {
		itemType string
		index    int
	}
	var items []viewItem
	for i := range data.LLMCalls {
		items = append(items, viewItem{"llm", i})
	}
	for i := range data.Modules {
		items = append(items, viewItem{"module", i})
	}
	for i := range data.CodeExecs {
		items = append(items, viewItem{"code", i})
	}
	for i := range data.ToolCalls {
		items = append(items, viewItem{"tool", i})
	}

	currentIdx := 0
	expanded := !cfg.Compact

	// Helper to print current item
	printItem := func(idx int, expanded bool) {
		item := items[idx]

		// Status bar
		fmt.Printf("%s%s Event %d/%d (%s) %s", BgBlue, White, idx+1, len(items), item.itemType, Reset)
		fmt.Printf(" %s[e]xpand [s]tats [?]help [q]uit%s\n", Dim, Reset)
		fmt.Println(strings.Repeat("─", 60))

		// Print the item
		switch item.itemType {
		case "llm":
			PrintLLMCall(data.LLMCalls[item.index], item.index, !expanded, "")
		case "module":
			PrintModule(data.Modules[item.index], !expanded, "")
		case "code":
			PrintCodeExec(data.CodeExecs[item.index], !expanded, "")
		case "tool":
			PrintToolCall(data.ToolCalls[item.index], !expanded)
		}

		// Navigation hints
		fmt.Printf("\n%s← k/↑  j/↓ →  g=first G=last%s\n", Dim, Reset)
	}

	ClearScreen()
	printItem(currentIdx, expanded)

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			break
		}

		switch n {
		case 1:
			switch buf[0] {
			case 'q', 3: // q or Ctrl+C
				ClearScreen()
				return
			case 'j':
				if currentIdx < len(items)-1 {
					currentIdx++
				}
			case 'k':
				if currentIdx > 0 {
					currentIdx--
				}
			case 'g':
				currentIdx = 0
			case 'G':
				currentIdx = len(items) - 1
			case 'e':
				expanded = !expanded
			case 's':
				// Show stats
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				ClearScreen()
				PrintDetailedStats(data)
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			case '?':
				// Show help
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				ClearScreen()
				printDSPyInteractiveHelp()
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			}
		case 3:
			// Arrow keys
			if buf[0] == 27 && buf[1] == 91 {
				switch buf[2] {
				case 65: // Up
					if currentIdx > 0 {
						currentIdx--
					}
				case 66: // Down
					if currentIdx < len(items)-1 {
						currentIdx++
					}
				}
			}
		}

		ClearScreen()
		printItem(currentIdx, expanded)
	}
}

func printDSPyInteractiveHelp() {
	fmt.Printf(`
%s%s Interactive Mode Help (DSPy Format) %s

%sNavigation:%s
  j, ↓       Next event
  k, ↑       Previous event
  g          Go to first event
  G          Go to last event

%sDisplay:%s
  e          Toggle expand/compact mode
  s          Show detailed statistics

%sOther:%s
  ?          Show this help
  q, Ctrl+C  Quit

`, BoldCyan, "═══", Reset, Bold, Reset, Bold, Reset, Bold, Reset)
}
