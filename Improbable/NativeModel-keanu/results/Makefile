#all: results graphs


clean:
	rm -r *.dot *.pdf *.csv figure


results:
	Rscript -e "require(knitr) ; require(markdown) ;knit('ResultsSummary.Rmd', 'ResultsSummary.md'); markdownToHTML('ResultsSummary.md', 'ResultsSummary.html');"
	rm ResultsSummary.md
	open ResultsSummary.html
