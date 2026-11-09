PY ?= python

install:
	$(PY) -m pip install -e ".[llm,dev]"

data:
	$(PY) scripts/fetch_data.py

app:
	streamlit run app/streamlit_app.py

demo:
	$(PY) scripts/run_demo.py

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"
