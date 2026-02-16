# Feature Workflow Settings

## Task Guidelines

- Tasks should include unit tests for the code being added
- Dependencies should be added in the tasks that require them, not as separate tasks
- Prefer larger tasks that complete a full vertical slice over small atomic changes
- Each task should update relevant documentation

## Commands
### Testing
- All tests: uv run pytest
- With coverage: uv run pytest --cov=casual_memory --cov-report=html
- Specific file: uv run pytest tests/classifiers/test_pipeline.py -v

### Code quality
- Format: uv run black src/
- Lint: uv run ruff check src/
- Type Check: uv run mypy src/casual_memory/