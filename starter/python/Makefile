.PHONY: build run demo viewer test clean
build:
	python3 -m compileall -q src
run:
	python3 src/main.py
demo:
	python3 src/main.py --demo
viewer:
	python3 tools/viewer_bridge.py
test:
	python3 -m unittest discover -s tests -v
clean:
	find src tests tools -type d -name __pycache__ -exec rm -rf {} +
