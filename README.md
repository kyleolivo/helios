# Helios

An AI agent

## Project Status

🚧 **Currently in development**

### Progress
- ✅ Project setup with modern Python tooling (uv, ruff, Pydantic)
- 🔄 LLM integration with Open Router API (in progress)
- ⏳ CLI chat interface
- ⏳ ReAct agent with tool capabilities
- ⏳ Documentation and portfolio polish

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Open Router API key (get one at [openrouter.ai/keys](https://openrouter.ai/keys))

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd helios
```

2. Install dependencies using `uv`:
```bash
# Install uv if you haven't already
brew install uv

# Sync dependencies
uv sync
```

3. Configure your environment:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Open Router API key
# Get a free key from: https://openrouter.ai/keys
```

4. Test the LLM connection:
```bash
uv run python test_llm_connection.py
```

If everything is set up correctly, you should see a conversation with the AI!

## Project Structure

```
helios/
├── src/helios/          # Main package
│   ├── core/           # Core agent logic
│   │   ├── llm.py      # LLM client abstraction
│   │   └── types.py    # Pydantic data models
│   ├── tools/          # Agent tools (coming soon)
│   └── utils/          # Configuration and utilities
├── tests/              # Test suite
├── examples/           # Example scripts
└── docs/              # Documentation
```

## What is Helios?

Helios is an educational project for learning AI agent development. Rather than using high-level frameworks immediately, this project builds core agent capabilities from scratch to develop deep understanding of:

- LLM API integration
- Prompt engineering
- ReAct (Reasoning + Acting) pattern
- Tool/function calling
- Agent orchestration loops

## Technology Stack

- **Python 3.11+** - Modern Python features
- **uv** - Fast dependency management
- **Open Router** - Unified API for multiple LLM providers
- **Pydantic** - Data validation and settings management
- **Click** - CLI framework (coming soon)
- **Rich** - Beautiful terminal output (coming soon)