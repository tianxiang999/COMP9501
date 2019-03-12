

all: main.pdf

main.pdf: main.tex
	pdflatex main.tex
	pdflatex main
	pdflatex main

.PHONY: clean

clean:
	rm -rf main.pdf
	rm -rf main.aux
	rm -rf main.log

