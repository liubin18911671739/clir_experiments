# Contributing to CLIR Experiments

Thank you for your interest in contributing to the Cross-Lingual IR Experiments toolkit!

## Development Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Development tools
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Formatting

Format code with Black:
```bash
black scripts/
```

Check style with flake8:
```bash
flake8 scripts/ --max-line-length=100
```

## Adding New Features

### Adding a New Dense Retrieval Model

1. **Update configuration** in `config/neuclir.yaml`:
   ```yaml
   dense:
     new_model:
       model_name: "model/path"
       batch_size: 128
       # ... other parameters
   ```

2. **Create encoder wrapper** (if needed) in a new file like `scripts/encoders_custom.py`

3. **Add build logic** to `scripts/build_index_dense.py`:
   ```python
   def build_new_model_index(config, lang, repo_root):
       # Implementation here
       pass
   ```

4. **Create search script** like `scripts/run_dense_new_model.py`

5. **Update README** with usage examples

### Adding a New Reranking Model

1. **Update configuration** in `config/neuclir.yaml`:
   ```yaml
   reranking:
     new_reranker:
       model_name: "model/path"
       # ... parameters
   ```

2. **Create reranker class** in `scripts/rerank_mt5.py` or a new file:
   ```python
   class NewReranker:
       def __init__(self, model_name, device='cuda'):
           # Initialize model
           pass

       def score_pairs(self, query_doc_pairs):
           # Score implementation
           pass
   ```

3. **Add to rerank script** command-line options

4. **Document** in README

## Testing

### Unit Tests

Create tests in `tests/` directory:

```python
# tests/test_utils_io.py
from scripts.utils_io import load_yaml

def test_load_yaml():
    config = load_yaml('config/neuclir.yaml')
    assert 'languages' in config
    assert 'dense' in config
```

Run tests:
```bash
pytest tests/
```

### Integration Tests

Test full pipelines:

```bash
# Build small test index
python scripts/build_index_dense.py \
    --config config/neuclir.yaml \
    --model mdpr \
    --lang test

# Verify index was created
ls indexes/dense/mdpr_test/
```

## Utility Functions

When adding utilities, follow these guidelines:

### utils_io.py

For I/O operations:
- File reading/writing
- Path resolution
- TREC format handling

### utils_topics.py

For topic/query handling:
- Topic file parsing
- Query preprocessing
- Qrels loading

### New utility modules

Create focused modules for specific tasks:
- `utils_eval.py` - Evaluation helpers
- `utils_preprocessing.py` - Text preprocessing
- etc.

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Longer description with more details if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### README Updates

When adding features:
1. Add to **Features** section
2. Add usage example in **Quick Start**
3. Add detailed docs in **Advanced Usage**
4. Update configuration section if needed

## Pull Request Process

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/new-model-support
   ```

2. **Make changes** following code style guidelines

3. **Test thoroughly**:
   - Run existing tests: `pytest tests/`
   - Test your new feature manually
   - Verify documentation

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add support for new dense retrieval model XYZ"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/new-model-support
   ```
   Then create a pull request on GitHub

6. **PR description should include**:
   - What feature/fix this adds
   - How to test it
   - Example usage
   - Any breaking changes

## Questions?

- Open an issue for bug reports or feature requests
- Start a discussion for questions about extending the toolkit
- Check existing issues and PRs before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
