# Contributing

This is an educational MLOps project, but changes should still be small,
reproducible, and easy to review.

## Local Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local environment settings:

```bash
cp .env.example .env
```

On macOS with Homebrew OpenJDK, set:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
```

## Common Commands

Run fast API tests:

```bash
make test
```

Run the full local ML pipeline:

```bash
make pipeline
```

Start the API:

```bash
make run-api
```

Start Docker services:

```bash
make docker-up
```

Stop Docker services:

```bash
make docker-down
```

## Development Workflow

1. Keep each change focused on one idea.
2. Add or update tests when API behavior changes.
3. Update `README.md` when commands, endpoints, or project behavior changes.
4. Run `make test` before committing.
5. Use the Swagger UI at `http://127.0.0.1:8001/docs` for manual API checks.

## Commit Style

Use short conventional commit messages:

```text
feat: add typed similar movies response
fix: align training data file names
docs: add api demo flow
test: cover stats endpoint response schema
chore: add docker build ignore rules
ci: split fast tests and ml pipeline smoke
```

## Pull Request Checklist

- Fast tests pass with `make test`.
- The API still opens at `http://127.0.0.1:8001/docs`.
- New or changed endpoints have response models.
- README or docs are updated when behavior changes.
- No local secrets, `.env`, virtual environments, or MLflow runs are committed.
