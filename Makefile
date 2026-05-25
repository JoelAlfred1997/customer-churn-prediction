.PHONY: install format lint test app clean

install:
	pip install -r requirements.txt
	pip install -e .

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	mypy src

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
