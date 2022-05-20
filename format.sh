poetry run python -m pytest tests
poetry run black .
poetry run isort . --profile black
poetry run flake8 .
# poetry run bandit .
# poetry run safety check