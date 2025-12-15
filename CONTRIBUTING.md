# Contributing to Zendesk Ticket Intelligence (ZTI)

Thank you for your interest in contributing to ZTI!

## Development Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Access to GPU machine for testing (optional for local development)

### Local Development
```bash
# Clone the repository
git clone --recursive https://github.com/djodom1134/zendesk-ticket-intelligence.git
cd zendesk-ticket-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Remote GPU Development
See [docs/remote-development.md](docs/remote-development.md) for SSH setup.

## Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Run Linting
```bash
ruff check .
ruff format .
```

### 5. Submit Pull Request
- Fill out the PR template completely
- Link to related issues
- Wait for CI to pass

## Code Style

- Use `ruff` for linting and formatting
- Type hints are required for all functions
- Follow PEP 8 conventions
- Write descriptive commit messages

## Testing on GPU Machine

Integration tests that require GPU should be marked:
```python
@pytest.mark.gpu
def test_ollama_inference():
    ...
```

Run GPU tests on the remote machine:
```bash
pytest tests/ -v -m gpu
```

## Questions?

Open an issue or reach out to the maintainers.

