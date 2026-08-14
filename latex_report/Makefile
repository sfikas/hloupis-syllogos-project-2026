TARGET=report_hloupis1
PDFREADER=evince
COMPILER=xelatex
all:
	$(COMPILER) $(TARGET).tex
#	$(COMPILER) $(TARGET).tex
#	bibtex $(TARGET).aux
#	$(COMPILER) $(TARGET).tex
show:
	$(PDFREADER) $(TARGET).pdf
clean:
	rm *.pdf *.log *.bbl *.aux *.blg *.fls *.toc *.fdb_latexmk *.*~ *.ist
