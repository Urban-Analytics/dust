library(nlme)
#library(ggplot2)
#library(docstring)

root.dir <- "~/dust/Projects/ABM_DA/stationsim/RK_Validation/RK_csvs"

load_RK_Data<- function(width, height, pop_total, gate_speed){
  
  #'load csv containing two groups of control and test RK trajectories
  #'
  #'@param width stationsim model width
  #'@param height stationsim model height
  #'@param pop_total number of agent in model
  #'@param gate_speed rate at which agent enter the model
  #'@return loaded csv
  
  setwd(root.dir)
  
  file<- sprintf("joint_%s_%s_%s_%s.csv", width, height, pop_total,
                 gate_speed)
  data <- read.csv(file)
  data<- groupedData(y ~ x | ids, data) 
  setwd(root.dir)
  
  #switch back to notebook directory to keep notebook happy.
  return(data)
}

compare_RK_Groups <- function(data, verbose = TRUE){
  
  #'Test if two groups of RK trajectories are statistically distinct
  #'
  #' Through trial and error we have found a second order (x^2) regression
  #' to provide most acceptable results given no grouping.
  #' 
  #' To determine if there is separation between the groups we do the following:
  #' 
  #' First build a saturated model (mmod123) with all possible terms assuming
  #' there is a difference between the groups and hence three additional
  #' split interaction terms between the group id (0 or 1) and each fixed effect
  #' (1, x , x^2).
  #' 
  #' For the full saturate model we have three level terms (gamma_0, gamma_1, 
  #' gamma_2). Taking every possible subset of gammas we have 8 possible models
  #' (NULL, 1, 2, 3, 12, 13, 23, 123). We wish to test if the null model performs 
  #' best. If it does, we have no evidence to suggest significant level effects
  #' and conclude no evidence of group separation
  #' 
  #' We test if the null model (mmod_null) peforms best out of all 8
  #' models using 7 likelihood ratio tests. Each likelihood test has a null
  #' hypothesis to accept the simpler reduced (null) model.
  #' 
  #' If any of the tests produces a significant p-value <0.05 we conclude 
  #' the null model does not perform best and there is evidence of group separation
  #' 
  #' If no p-values are significant we conclude the null model performs best
  #' and we have no evidence of group separation.
  #' 
  #' NOTE: It may be better to use stepwise regression based on AIC here.
  #' (see cAIC4 package)
  #' Printing the full anova lets one decide either way. 
  #' The lowest AIC score is often taken as the best performing model. 
  #' If null model has lowest AIC assume no evidence for group difference.
  #' @param data some loaded csv containing control/test group RK data
  #' @param verbose print full anova outputs? recommended.
  #' @return an1 anova with 4 models to compare
  
  #data scaling due to large difference in magnitude
  data["x"] <- scale(data["x"])
  data["y"] <- scale (data["y"])
  
  #all 8 combinations of the three factor effects
  #numbers 1,2,3 indicate gamma_0, gamma_1, or gamma_2 are 
  #present respectively.
  #note required ML method over default REML due to different fixed effects.
  
  mmod123<- lme(y ~ I(x**2)*factor(split) + x*factor(split) ,
                      data=data, random =  ~ x |ids, method = "ML" )
  
  mmod12<- lme(y ~ I(x**2)+ x*factor(split) ,
                data=data, random =  ~ x |ids, method = "ML" )
  mmod13<- lme(y ~ I(x**2)*factor(split) + x ,
               data=data, random =  ~ x |ids, method = "ML" )
  mmod23<- lme(y ~ I(x**2)*factor(split) + x*factor(split) - factor(split) ,
               data=data, random =  ~ x |ids, method = "ML" )
  
  mmod1<- lme(y ~ I(x**2) + x + split ,
              data=data, random =  ~ x|ids, method = "ML")
  mmod2<- lme(y ~ I(x**2) + x *factor(split) - factor(split),
              data=data, random =  ~ x|ids, method = "ML")
  mmod3<- lme(y ~ I(x**2)*factor(split) + x - factor(split),
              data=data, random =  ~ x|ids, method = "ML")
  
  mmod_null<-  lme(y ~ I(x**2) + x ,
               data=data, random =  ~ x|ids, method = "ML")
  

  an123 <- anova(mmod123, mmod_null)
  an12 <- anova(mmod12, mmod_null)
  an13 <- anova(mmod13, mmod_null)
  an23 <- anova(mmod23, mmod_null)
  an1 <- anova(mmod1, mmod_null)
  an2 <- anova(mmod2, mmod_null)
  an3 <- anova(mmod3, mmod_null)
  
  if(verbose == TRUE){
  print(an123)
  print(an12)
  print(an13)
  print(an23)
  print(an1)
  print(an2)
  print(an3)
  }
  
  ps <- c(an123$`p-value`[2],  an12$`p-value`[2], an13$`p-value`[2],
          an23$`p-value`[2],  an1$`p-value`[2], an2$`p-value`[2],
          an3$`p-value`[2])
  return(ps)
}

spaghetti_plot <- function(data){
  
  #' R version of spaghetti plots for panel data.
  #' This version is much easier to implement than the python one 
  #' and is here incase it becomes more convenient later
  #' @param data some loaded csv containing control/test group RK data
  
  s_plot<- ggplot(data = data, mapping = aes(x = x, y = y,
                                    group = ids)) + 
            geom_line(aes(col = factor(split))) 
  print(s_plot)
}

main <- function(width, height, pop_total, gate_speed){
  
  #' main function for comparing two groups of ripley's K stationsim curves
  #' load data
  #' fit models with and without difference in group effects
  #' determine which model is preferable based on anova between two models.
  #'@param width stationsim model width
  #'@param height stationsim model height
  #'@param pop_total number of agent in model
  #'@param gate_speed rate at which agent enter the model
  #'@return an1 anova with 4 models to compare
  
  data<- load_RK_Data(width, height, pop_total, gate_speed)
  ps <- compare_RK_Groups(data)
  print(ps)
  #spaghetti_plot(data)
  return(ps)
}

width <- 200
height <- 50
pop_total <- 30 
gate_speed <- 1

ps <- main(width, height, pop_total, gate_speed)
