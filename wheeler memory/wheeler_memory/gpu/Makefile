# Wheeler Memory — GPU kernel build
#
# Usage:
#   make              Build libwheeler_ca.so
#   make clean        Remove build artifacts
#   make test         Verify .so exports ca_evolve_* symbols
#   make verify       Run bench_gpu.py --verify-only (CPU vs GPU correctness)

# Auto-detect installed GPU arch; fall back to gfx1201 (RDNA4 / RX 9070)
GPU_ARCH ?= $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1)
ifeq ($(GPU_ARCH),)
    GPU_ARCH := gfx1201
endif

HIPCC  ?= hipcc
CFLAGS  = --offload-arch=$(GPU_ARCH) -O3 -shared -fPIC
TARGET  = libwheeler_ca.so
SRC     = ca_kernel.hip

.PHONY: all clean test verify

all: $(TARGET)
	@echo "Built $(TARGET) for $(GPU_ARCH)"

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "=== Library: $(TARGET) ==="
	@ls -lh $(TARGET)
	@echo "=== Exported symbols ==="
	@nm -D $(TARGET) | grep "ca_evolve" || (echo "ERROR: symbols missing"; exit 1)
	@echo "OK"

verify: $(TARGET)
	@echo "=== CPU vs GPU correctness ==="
	cd ../.. && python scripts/bench_gpu.py --verify-only
