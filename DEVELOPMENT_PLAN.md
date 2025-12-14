# Helios Development Plan

## Overview
Build an AI agent from scratch to understand agent fundamentals while strengthening Python skills. Focus on iterative development with each phase introducing new agent concepts.

## Current Status
✅ **Foundation Complete**
- Project scaffolding with modern Python tooling (uv, ruff, mypy)
- LLM abstraction layer (OpenRouterLLM with streaming support)
- Pydantic data models (Message, Conversation, Settings)
- Comprehensive test suite (54 tests, 100% coverage)
- Configuration management with environment variables

## Learning Goals
1. **Agent Fundamentals**: Understand core agent patterns (ReAct, tool use, memory)
2. **Python Best Practices**: Type hints, testing, async patterns, CLI design
3. **LLM Integration**: Streaming, context management, prompt engineering
4. **Iterative Development**: Build features incrementally, test-driven approach

## Development Phases

### Phase 1: Basic CLI Chat Interface (Next Up)
**Goal**: Interactive chat loop to understand conversation flow and state management

**Learning Focus**:
- CLI design with Click
- Managing conversation state
- User interaction patterns
- Rich terminal output

**Tasks**:
1. Create `src/helios/cli.py` with basic Click commands
2. Implement interactive chat loop
3. Add conversation history display
4. Support streaming responses with visual feedback
5. Add tests for CLI logic (conversation management, not UI)

**Files to Create/Modify**:
- `src/helios/cli.py` - Main CLI entry point
- `src/helios/core/chat.py` - Chat session management
- `tests/test_chat.py` - Chat logic tests

**Success Criteria**:
- `helios chat` starts an interactive session
- Messages stream to terminal in real-time
- Conversation history persists during session
- Clean exit with Ctrl+C or /exit command

---

### Phase 2: Basic Tool Calling
**Goal**: Understand how agents use tools to extend capabilities

**Learning Focus**:
- Tool/function calling patterns
- JSON schema for tool definitions
- Tool execution and result handling
- Error handling for tool failures

**Tasks**:
1. Create tool abstraction (`Tool` base class)
2. Implement simple tools (calculator, datetime, web search)
3. Add tool registry and discovery
4. Integrate tool calling into LLM conversation flow
5. Add tool-specific tests

**Files to Create/Modify**:
- `src/helios/tools/base.py` - Tool base class
- `src/helios/tools/calculator.py` - Simple calculator tool
- `src/helios/tools/web_search.py` - DuckDuckGo search tool
- `src/helios/tools/registry.py` - Tool registry and management
- `src/helios/core/llm.py` - Add tool calling support
- `tests/test_tools.py` - Tool tests

**Success Criteria**:
- Agent can call tools when needed
- Tools return structured results
- Errors handled gracefully
- User sees when tools are being used

---

### Phase 3: ReAct Agent Pattern
**Goal**: Implement the ReAct (Reasoning + Acting) pattern for multi-step problem solving

**Learning Focus**:
- ReAct loop: Thought → Action → Observation → repeat
- Chain-of-thought reasoning
- Multi-turn agent interactions
- When to stop reasoning (max iterations, goal reached)

**Tasks**:
1. Create ReAct agent class
2. Implement thought/action/observation loop
3. Add prompt templates for ReAct reasoning
4. Add iteration limits and stopping conditions
5. Log reasoning traces for debugging
6. Add tests for ReAct logic

**Files to Create/Modify**:
- `src/helios/core/agent.py` - ReAct agent implementation
- `src/helios/core/prompts.py` - Prompt templates
- `src/helios/core/types.py` - Add ReAct-specific types (Thought, Action, Observation)
- `tests/test_agent.py` - Agent reasoning tests

**Success Criteria**:
- Agent can break down complex queries
- Reasoning steps visible to user
- Agent uses tools when appropriate
- Agent stops when goal is reached

---

### Phase 4: Memory and Context Management
**Goal**: Handle long conversations and maintain context effectively

**Learning Focus**:
- Context window management
- Conversation summarization
- Selective history (what to keep/discard)
- Vector embeddings for semantic search (optional advanced topic)

**Tasks**:
1. Implement token counting
2. Add conversation summarization when approaching limits
3. Create message prioritization (system > recent > relevant)
4. Add conversation persistence (save/load from disk)
5. Add tests for memory management

**Files to Create/Modify**:
- `src/helios/core/memory.py` - Memory management
- `src/helios/core/types.py` - Add ConversationHistory type
- `src/helios/utils/tokens.py` - Token counting utilities
- `tests/test_memory.py` - Memory tests

**Success Criteria**:
- Conversations don't exceed context limits
- Important context preserved during summarization
- Conversations can be saved and resumed
- User can see when summarization happens

---

### Phase 5: Advanced Tool Orchestration
**Goal**: Compose tools and handle complex multi-tool workflows

**Learning Focus**:
- Tool chaining
- Parallel tool execution
- Tool dependencies
- Handling tool failures and retries

**Tasks**:
1. Add tool dependency tracking
2. Implement parallel tool execution
3. Add tool result caching
4. Create tool composition patterns
5. Add workflow visualization

**Files to Create/Modify**:
- `src/helios/tools/orchestrator.py` - Tool orchestration
- `src/helios/tools/base.py` - Add dependency metadata
- `tests/test_orchestration.py` - Orchestration tests

**Success Criteria**:
- Agent can chain tool calls efficiently
- Independent tools run in parallel
- Failed tools trigger retries or alternatives
- User sees tool execution progress

---

### Phase 6: Polish and Optimization
**Goal**: Production-ready features and performance optimization

**Learning Focus**:
- Async patterns in Python
- Caching strategies
- Error recovery patterns
- User experience refinements

**Tasks**:
1. Add async/await for LLM and tool calls
2. Implement response caching
3. Add retry logic with exponential backoff
4. Improve error messages and help text
5. Add configuration presets (different agent personalities)
6. Performance profiling and optimization

**Files to Create/Modify**:
- `src/helios/core/llm.py` - Add async methods
- `src/helios/tools/base.py` - Add async tool support
- `src/helios/utils/cache.py` - Response caching
- `src/helios/utils/retry.py` - Retry logic
- Multiple files - Add async throughout

**Success Criteria**:
- All I/O operations are async
- Responses cached appropriately
- Failures handled gracefully with retries
- Smooth user experience

---

## Development Principles

### Test-Driven Development
- Write tests before or alongside features
- Maintain 90%+ coverage
- Test behavior, not implementation details
- Use mocking for external services (LLM, web search)

### Incremental Complexity
- Each phase builds on previous work
- Features can be used independently
- Clear separation of concerns
- Refactor as patterns emerge

### Learning-Focused
- Document design decisions and tradeoffs
- Comment non-obvious patterns
- Keep examples up to date
- Note interesting Python techniques used

### Professional Quality
- Type hints everywhere
- Comprehensive docstrings
- Clean commit history with conventional commits
- CI/CD for automated testing (future)

---

## Key Technical Decisions

### Why OpenRouter?
- Access to multiple LLM providers through one API
- Compatible with OpenAI SDK (familiar interface)
- Cost-effective for experimentation

### Why Build from Scratch vs Framework?
- Deep understanding of agent fundamentals
- Learn by doing, not by configuration
- Full control over behavior and architecture
- Foundation for understanding frameworks later (LangChain, AutoGen, etc.)

### Why Pydantic?
- Type safety with runtime validation
- Clean data modeling
- Easy serialization for API calls
- Industry standard for Python APIs

### Why Click + Rich?
- Click: Battle-tested CLI framework
- Rich: Beautiful terminal output without complexity
- Good learning examples in their docs
- Professional-looking UX

---

## Next Immediate Steps

1. **Phase 1 Sprint**: Build the basic CLI chat interface
   - Start with simple REPL loop
   - Add streaming display
   - Test with real LLM calls
   - Iterate on UX

2. **Questions to Answer**:
   - Should conversation history show in terminal or separate pane?
   - What commands should be supported? (/clear, /save, /help, etc.)
   - Should we support multiple conversation sessions?

3. **After Phase 1**:
   - Review and refactor based on learnings
   - Update this plan with insights
   - Begin Phase 2: Tool calling

---

## Resources for Learning

- **ReAct Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **Tool Use**: OpenAI function calling docs (patterns apply to other providers)
- **Python Async**: Real Python's async/await tutorial
- **Agent Patterns**: LangChain docs (for patterns, not necessarily using the library)

---

## Success Metrics

- ✅ Working CLI chat interface
- ✅ Agent can use at least 3 different tools
- ✅ ReAct reasoning visible and functional
- ✅ Conversations can handle 10+ turns
- ✅ Test coverage above 90%
- ✅ Code passes type checking (mypy)
- ✅ Clean, documented codebase ready for portfolio

---

## Future Possibilities (Beyond Initial Scope)

- Web interface (FastAPI + React)
- Multi-agent collaboration
- Custom tool creation via config
- Fine-tuned models for specific tasks
- Agent benchmarking and evaluation
- Deployment to cloud (AWS Lambda, etc.)
