library(nlme)
library(ggplot2)
library(docstring)


load_RK_Data<- function(width, height, pop_total, gate_speed){
  
  #'load csv of two groups of RK trajectories
  file<- sprintf("joint_%s_%s_%s_%s.csv", width, height, pop_total, gate_speed)
  setwd("~/dust/Projects/ABM_DA/stationsim/RK_csvs")
  data <- read.csv(file)
  data<- groupedData(y ~ x | ids, data) 
  return(data)
}

compare_RK_Groups <- function(data){
#'Test if two groups of RK trajectories are statistically indistinguishable
#
  
#saturated model assuming difference between groups
#note required ML method over default REML due to different fixed effects
mmod0<- lme(y ~ I(x**2)*factor(split) + x*factor(split) ,
                    data=data, random =  ~ x |ids, method = "ML" )
#alternate hypothesis of no difference between groups
mmod1<- lme(y ~ I(x**2) + x ,
            data=data, random =  ~ x|ids, method = "ML")

an1 =anova(mmod0, mmod1)
p = an1$`p-value`[2]

return(an1)

}

spaghetti_plot <- function(data){
  
  s_plot<- ggplot(data = data, mapping = aes(x = x, y = y,
                                    group = ids)) + 
            geom_line(aes(col = factor(split))) 
  print(s_plot)
}

main <- function(width, height, pop_total, gate_speed){
  data<- load_RK_Data(width, height, pop_total, gate_speed)
  an1 <- compare_RK_Groups(data)
  spaghetti_plot(data)
  p<- an1$`p-value`[2]
  
  if(p<0.05){
    print("Reject NH. Evidence for difference between groups.")
  } else{
    print("Accept NH. No evidence for difference between groups.")
  }
  return(an1)
}

width <- 200
height <- 50
pop_total <- 10 
gate_speed <- 1 

an1 <- main(width, height, pop_total, gate_speed)

