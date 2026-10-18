CMAKE      ?= cmake
BUILD_TYPE ?= Release
JOBS       ?= 4
BUILD_DIR  ?= build

all: build

init:
	git submodule update --init --recursive

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)