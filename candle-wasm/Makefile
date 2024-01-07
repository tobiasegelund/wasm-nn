.PHONY: build move

run: build move

build:
	wasm-pack build --target web

move:
	rm -rf ../react-wasm/pkg
	cp -R pkg ../react-wasm/