PYTHON ?= python3

.PHONY: all run validate test serve clean

all: run validate

run:
	$(PYTHON) run.py

validate:
	$(PYTHON) validate.py

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

serve:
	$(PYTHON) serve.py

clean:
	rm -rf artifacts
