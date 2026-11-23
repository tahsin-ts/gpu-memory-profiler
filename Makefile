.PHONY: test demo docker
test:
	python -m pytest -q
demo:            ## the fragmentation OOM and its fix, then the analysis
	python -m gpumem simulate --capacity-gib 6 --snapshot /tmp/frag.pkl || true
	python -m gpumem simulate --capacity-gib 6 --max-split-mb 128
	python -m gpumem analyze /tmp/frag.pkl
docker:
	docker build -t gpumem:1.0.0 .
