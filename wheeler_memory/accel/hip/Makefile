# Wheeler Memory — Unified HIP kernel build
#
# Usage:
#   make              Build all available kernels
#   make ca           Build CA evolution kernel (v2, variable grid)
#   make ca-v1        Build CA evolution kernel (v1, 64x64 fixed, legacy)
#   make encode       Build encoding kernel (hippocampus + context-RI)
#   make similarity   Build similarity search kernel
#   make cortex       Build cortex settlement kernel
#   make clean        Remove all .so files
#   make test         Verify exported symbols
#
# Kernel naming: hip_<function>.hip → libwheeler_<function>.so

# Auto-detect installed GPU arch; fall back to gfx1201 (RDNA4 / RX 9070 XT)
GPU_ARCH ?= $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1)
ifeq ($(GPU_ARCH),)
    GPU_ARCH := gfx1201
endif

HIPCC  ?= hipcc
CFLAGS  = --offload-arch=$(GPU_ARCH) -O3 -shared -fPIC

# Only build what has source files
SOURCES := $(wildcard ca_evolve.hip ca_evolve_v1.hip hip_encode.hip hip_similarity.hip hip_cortex.hip)
TARGETS :=

ifneq ($(filter ca_evolve.hip,$(SOURCES)),)
    TARGETS += libwheeler_ca.so
endif
ifneq ($(filter ca_evolve_v1.hip,$(SOURCES)),)
    TARGETS += libwheeler_ca_v1.so
endif
ifneq ($(filter hip_encode.hip,$(SOURCES)),)
    TARGETS += libwheeler_encode.so
endif
ifneq ($(filter hip_similarity.hip,$(SOURCES)),)
    TARGETS += libwheeler_similarity.so
endif
ifneq ($(filter hip_cortex.hip,$(SOURCES)),)
    TARGETS += libwheeler_cortex.so
endif

.PHONY: all ca ca-v1 encode similarity cortex clean test

all: $(TARGETS)
	@echo "Built $(TARGETS) for $(GPU_ARCH)"

ca: libwheeler_ca.so
ca-v1: libwheeler_ca_v1.so
encode: libwheeler_encode.so
similarity: libwheeler_similarity.so
cortex: libwheeler_cortex.so

libwheeler_ca.so: ca_evolve.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

libwheeler_ca_v1.so: ca_evolve_v1.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

libwheeler_encode.so: hip_encode.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

libwheeler_similarity.so: hip_similarity.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

libwheeler_cortex.so: hip_cortex.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f libwheeler_*.so

test: $(TARGETS)
	@echo "=== Checking exported symbols ==="
	@for lib in $(TARGETS); do \
		echo "--- $$lib ---"; \
		nm -D $$lib | grep -E "ca_evolve|encode|similarity|cortex" || echo "  (no matching symbols)"; \
	done
	@echo "OK"
