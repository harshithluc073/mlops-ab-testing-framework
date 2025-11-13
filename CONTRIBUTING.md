# Contributing to MLOps A/B Testing Framework

First off, thank you for considering contributing to the MLOps A/B Testing Framework! It's people like you that make this tool better for everyone.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, configuration files)
- **Describe the behavior you observed** and what you expected
- **Include screenshots** if applicable
- **Specify your environment**: OS, Python version, package versions

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **Provide examples** of how it would be used

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** and ensure they follow the coding standards
4. **Add tests** for any new functionality
5. **Ensure all tests pass**: `pytest`
6. **Update documentation** if needed
7. **Commit your changes** with clear, descriptive messages
8. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlops-ab-testing-framework.git
cd mlops-ab-testing-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **Black** for code formatting: `black .`
- Use **isort** for import sorting: `isort .`
- Run **flake8** for linting: `flake8`
- Add type hints where appropriate
- Maximum line length: 100 characters

### Documentation

- Write clear docstrings for all public modules, functions, classes, and methods
- Use **Google style** docstrings
- Update README.md if you change functionality
- Add comments for complex logic

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Run tests before submitting PR: `pytest`

### Commit Messages

- Use clear and meaningful commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add detailed description if needed

Example:
```
Add Bayesian inference for statistical analysis

- Implement beta-binomial conjugate prior
- Add credible interval calculation
- Include unit tests for edge cases
```

## Project Structure

```
mlops-ab-testing-framework/
├── src/mlops_ab_testing/    # Main package
│   ├── core/                # Core testing framework
│   ├── metrics/             # Metrics calculation
│   ├── statistics/          # Statistical analysis
│   ├── explainability/      # Model explainability
│   ├── visualization/       # Plotting and charts
│   ├── reporting/           # Report generation
│   ├── tracking/            # MLflow/W&B integration
│   ├── cli/                 # Command-line interface
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
└── configs/                 # Configuration templates
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mlops_ab_testing --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::test_accuracy_calculation
```

### Writing Tests

```python
import pytest
from mlops_ab_testing.metrics import calculate_accuracy

def test_accuracy_calculation():
    """Test accuracy metric calculation."""
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    
    accuracy = calculate_accuracy(y_true, y_pred)
    
    assert accuracy == 0.8
    assert 0 <= accuracy <= 1

def test_accuracy_with_empty_input():
    """Test accuracy with edge case."""
    with pytest.raises(ValueError):
        calculate_accuracy([], [])
```

## Documentation Guidelines

### Docstring Example

```python
def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for given data.
    
    Args:
        data: Array of numerical values
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Tuple containing (lower_bound, upper_bound)
    
    Raises:
        ValueError: If confidence level is not between 0 and 1
    
    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> lower, upper = calculate_confidence_interval(data)
        >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    pass
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a new tag: `git tag -a v0.1.0 -m "Release version 0.1.0"`
4. Push tags: `git push origin v0.1.0`
5. GitHub Actions will automatically build and publish to PyPI

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing! 🎉
