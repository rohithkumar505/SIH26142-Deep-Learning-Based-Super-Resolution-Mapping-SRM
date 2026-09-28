.PHONY: install test run api ui docker

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest test_app.py -v

run:
	. .venv/bin/activate && python app.py

ui:
	python3 -m http.server 8080

docker:
	docker compose up --build
