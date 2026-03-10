MARP := npx marp
PANDOC := pandoc
WEASYPRINT := $(shell command -v weasyprint 2>/dev/null || echo ../.venv/bin/weasyprint)
OUTDIR := output

SLIDE_SRCS := slides/01_investor_pitch.md slides/02_developer_pitch.md slides/03_general_pitch.md
SLIDE_PDFS := $(OUTDIR)/Wheeler_Memory_Investor_Pitch.pdf \
              $(OUTDIR)/Wheeler_Memory_Developer_Pitch.pdf \
              $(OUTDIR)/Wheeler_Memory_General_Pitch.pdf

.PHONY: all slides one-pager whitepaper demo clean

all: slides one-pager whitepaper demo

$(OUTDIR):
	mkdir -p $(OUTDIR)

# ── Slide decks (Marp) ──────────────────────────────────────────────

slides: $(SLIDE_PDFS)

$(OUTDIR)/Wheeler_Memory_Investor_Pitch.pdf: slides/01_investor_pitch.md | $(OUTDIR)
	$(MARP) --pdf --allow-local-files $< -o $@

$(OUTDIR)/Wheeler_Memory_Developer_Pitch.pdf: slides/02_developer_pitch.md | $(OUTDIR)
	$(MARP) --pdf --allow-local-files $< -o $@

$(OUTDIR)/Wheeler_Memory_General_Pitch.pdf: slides/03_general_pitch.md | $(OUTDIR)
	$(MARP) --pdf --allow-local-files $< -o $@

# ── One-pager (Marp, single slide) ──────────────────────────────────

one-pager: $(OUTDIR)/Wheeler_Memory_One_Pager.pdf

$(OUTDIR)/Wheeler_Memory_One_Pager.pdf: one_pager/darman_one_pager.md | $(OUTDIR)
	$(MARP) --pdf --allow-local-files $< -o $@

# ── Whitepaper (pandoc markdown → PDF) ──────────────────────────────

whitepaper: $(OUTDIR)/Wheeler_Memory_Whitepaper.pdf

$(OUTDIR)/Wheeler_Memory_Whitepaper.pdf: whitepaper/wheeler_memory_whitepaper.md style.css | $(OUTDIR)
	$(PANDOC) $< -o $@ --pdf-engine=$(WEASYPRINT) \
		--number-sections --toc --css=style.css

# ── Demo script (pandoc markdown → PDF) ─────────────────────────────

demo: $(OUTDIR)/Wheeler_Memory_Demo_Script.pdf

$(OUTDIR)/Wheeler_Memory_Demo_Script.pdf: demo_script/demo_script.md style.css | $(OUTDIR)
	$(PANDOC) $< -o $@ --pdf-engine=$(WEASYPRINT) --css=style.css

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -rf $(OUTDIR)/*.pdf
