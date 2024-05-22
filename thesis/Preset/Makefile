# nechajte iba jeden z main.pdf a main-en.pdf
all: main.pdf main-en.pdf

main-en.pdf: main-en.tex *.tex *.bib images/*
	pdflatex main-en
	bibtex main-en
	pdflatex main-en
	pdflatex main-en
