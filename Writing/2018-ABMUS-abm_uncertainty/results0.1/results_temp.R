library(ggplot2)   # For density scatter plot
library(reshape2)
setwd("/Users/nick/conferences/2018/ABMUS2018/abstract-nick/results0.1/")

df <- read.csv('results_temp.csv')

df <- melt(df ,  id.vars = 'Iteration', variable.name = 'sample')


ggplot(df, aes(x=Iteration,y=value)) +
  #geom_point(size=0.1, color="black")
  #geom_line(color="grey")+
  #geom_line(color=series)+
  geom_smooth(method="loess", se=TRUE, level=0.99, color="red")+
  ylab("Number of agents")+
  xlab("Iterations")+
  ggtitle(paste0("BNER",", ",e))

  
  
  
ggplot(data = data.frame("x"=x, "y"=y ), mapping = aes(x,y)) +
  geom_point(size=0.1, color="black") + 
  #geom_hex(bins=15, show.legend = FALSE) +
  geom_smooth(method="loess", se=TRUE, level=0.99 )+ #, color="red") +
  ylab(paste("Error (",e,")"))+
  xlab("Number of Cells")+
  ggtitle(paste0("BNER",", ",e))
plots[[plot.count]] <- the.plot
plot.count <- plot.count + 1



