# Wheeler Memory — GPU kernel build
#
# Usage:
#   make              Build libwheeler_ca.so
#   make clean        Remove build artifacts
#   make test         Quick sanity check

GPU_ARCH ?= gfx1201
HIPCC ?= hipcc
CFLAGS = -shared -fPIC -O3 --offload-arch=$(GPU_ARCH)
TARGET = libwheeler_ca.so
SRC = ca_kernel.hip

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Library built successfully: $(TARGET)"
	@ls -lh $(TARGET)
	@echo "Symbols:"
	@nm -D $(TARGET) | grep "ca_evolve"
