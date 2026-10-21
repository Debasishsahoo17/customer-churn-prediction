.PHONY: data train app test all

data:
	python src/generate_data.py

train:
	python src/train.py

app:
	streamlit run app.py

test:
	pytest -q

all: data train test
