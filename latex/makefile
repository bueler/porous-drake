all: doc.pdf

doc.pdf: doc.aux
	pdflatex doc

doc.aux: doc.tex
	pdflatex doc

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *.synctex.gz *.pyc *.bcf *.run.xml *~

