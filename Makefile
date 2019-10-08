.PHONY: clean install build run

NAMESPACE = advr

clean:
	rm -rf *.o
	rm -rf .cache
	rm -rf .eggs
	rm -rf *.egg-info
	rm -rf *.png
	rm -rf build

build:
	docker build -t $(NAMESPACE) .

run:
	docker run --name $(NAMESPACE) -p 8765:8765 -v $(shell pwd):/app $(NAMESPACE) || \
	docker stop $(NAMESPACE); docker rm $(NAMESPACE) && \
	docker run --name $(NAMESPACE) -p 8765:8765 -v $(shell pwd):/app $(NAMESPACE)

stop:
	docker stop $(NAMESPACE); docker rm $(NAMESPACE)

report:
	Rscript doc/render.R

lint: install
	python setup.py test
	pip install pytest
	pytest src --flake8
