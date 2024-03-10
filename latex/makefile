all: porous.pdf

porous.pdf: porous.aux
	pdflatex porous

porous.aux: porous.tex
	pdflatex porous

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *.synctex.gz *.pyc *~

