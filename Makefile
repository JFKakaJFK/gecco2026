.PHONY: all tidy iwyu fmt configure reconfigure build test docs clean

GENERATOR ?=
NPROC ?= 1
OS:=$(shell uname -s)

# use ninja if available
ifeq ($(GENERATOR),)
	ifneq ($(shell which ninja),)
		GENERATOR := Ninja
	else
    	GENERATOR := "Unix Makefiles"
	endif
endif

# use the number of CPUs to parallelize builds and tests
ifeq ($(OS),Linux)
	NPROC := $(shell grep -c ^processor /proc/cpuinfo)
endif
ifeq ($(OS),Darwin)
	NPROC := $(shell sysctl -n hw.ncpu)
endif

all: fmt configure build test

ci: clean tidy iwyu fmt configure build test bindings # docs pydocs

reconfigure: clean configure

# type ?= Release
type ?= Debug
configure:
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=$(type) -G $(GENERATOR)

build: configure
	@cmake --build build -j$(NPROC)

test: build
	# @CTEST_OUTPUT_ON_FAILURE=1 cmake --build build --target test -- -j$(NPROC)
	@cd build && ctest -j$(NPROC) --output-on-failure

fmt:
	@git ls-files -- '*.cpp' '*.h' ':!extern/*' \
	| xargs clang-format -i -style=file

tidy:
	cmake -S . -B build-tidy -DCMAKE_CXX_CLANG_TIDY="$(which clang-tidy);-fix"

# requires IWYU (e.g. brew install include-what-you-use)
iwyu:
	if [ -z "$(shell which include-what-you-use)" ]; then \
	  echo "include-what-you-use not found, skipping IWYU"; \
	else \
      cmake -S . -B build-iwyu -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=include-what-you-use \
      && cmake --build build-iwyu 2> iwyu.out \
      ; \
	fi
	# TODO at some point automatically fix the includes...

bindings:
	cp -f pylib/nanobind/bindings_empty.cpp pylib/nanobind/bindings.cpp \
	&& uv run scripts/autogenerate_bindings.py \
	&& uv pip install -v -e .\
	&& uv pip uninstall -v .

docs:
	@echo "TODO Generate the top-level docs"
	cmake --build build --target docs
	uvx python -m http.server 8080 --bind 127.0.0.1 --directory build/lib/docs/html

pydocs: bindings
	@echo "TODO Get the stubs to a point where they can be parsed without errors or use another documentation generator"
	uv run --with pdoc pdoc pygom -o build/pygom/docs
	uvx python -m http.server 8081 --bind 127.0.0.1 --directory build/pygom/docs

clean:
	@rm -rf build build-iwyu build-tidy
