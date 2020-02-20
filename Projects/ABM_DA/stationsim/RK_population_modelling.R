library(nlme)
library(ggplot2)
library(docstring)


load_RK_Data<- function(width, height, pop_total, gate_speed){
  
  #'load csv containing two groups of control and test RK trajectories
  #'
  #'@param width stationsim model width
  #'@param height stationsim model height
  #'@param pop_total number of agent in model
  #'@param gate_speed rate at which agent enter the model
  #'@return loaded csv
  
  file<- sprintf("joint_%s_%s_%s_%s.csv", width, height, pop_total,
                 gate_speed)
  setwd("~/dust/Projects/ABM_DA/stationsim/RK_csvs")
  data <- read.csv(file)
  data<- groupedData(y ~ x | ids, data) 
  return(data)
}

compare_RK_Groups <- function(data){
  
  #'Test if two groups of RK trajectories are statistically distinct
  #'
  #' Through trial and error we have found a second order (x^2) regression
  #' to provide most acceptable results given no grouping.
  #' 
  #' To determine if there is separation between the groups
  #' we perform a backwards stepwise regression here as follows:
  #' 
  #' First build a saturated model (mmod1) with all possible terms assuming
  #' there is a difference between the groups and hence three additional
  #' interaction terms between the group id (0 or 1) and each fixed effect
  #' (1, x , x^2).
  #' 
  #' We then remove the three additional terms one at a time fitting a model 
  #' for each case. This gives us 4 models with interaction coefficients 
  #' between fixed effects for (1, x, x^2), (1, x), (1) and no terms
  #' respectively (mmod2, mmod3 and mmod4).
  #' 
  #' We test if the reduced model (mmod4) peforms best out of all 4
  #' models and thus determine if there is any evidence to believe the
  #' two groups of trajectories are statistically distinct. 
  #' 
  #' We test this using R's anova function taking the model with the minimum 
  #' Aikake's Information Criterion (AIC) to be the best performing model.
  #' If mmod4 has the lowest AIC we assume it performs best and thus
  #' there is no evidence of separation between the groups.
  #' @param data some loaded csv containing control/test group RK data
  #' @return an1 anova with 4 models to compare
  
  #saturated model assuming difference between groups
  #note required ML method over default REML due to different fixed effects
  mmod1<- lme(y ~ I(x**2)*factor(split) + x*factor(split) ,
                      data=data, random =  ~ x |ids, method = "ML" )
  #alternate hypothesis of no difference between groups
  mmod2<- lme(y ~ I(x**2) + x *factor(split) ,
              data=data, random =  ~ x|ids, method = "ML")
  mmod3<- lme(y ~ I(x**2) + x + factor(split),
              data=data, random =  ~ x|ids, method = "ML")
  mmod4<-  lme(y ~ I(x**2) + x ,
               data=data, random =  ~ x|ids, method = "ML")
  an1 =anova(mmod1, mmod2, mmod3, mmod4)
  p = an1$`p-value`[2]
  
  return(an1)
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

print_Results <- function(an){
  
  #' Print results from anova comparing saturated and reduced models
  #' Choose the model with smaller Aikake's Information Criterion (AIC).
  #' @param  an anova object with required AIC scores
  
    print(an)
    prefered <- which(an$AIC == min(an$AIC))
    print("Choose model with smallest AIC")
    print("AICs")
    print(an$AIC)
    if(prefered == 4){
      print("Prefer Reduced Model. No difference between groups")
    } else  {
        print("Prefer a Saturated Model. Significant difference between groups")
    }
}

main <- function(width, height, pop_total, gate_speed){
  
  #' main function for comparing two groups of ripley's K stationsim curves
  #' load data
  #' fit two models with and without difference in group effects
  #' determine which model is preferable based on anova between two models.
  #'@param width stationsim model width
  #'@param height stationsim model height
  #'@param pop_total number of agent in model
  #'@param gate_speed rate at which agent enter the model
  #'@return an1 anova with 4 models to compare
  
  data<- load_RK_Data(width, height, pop_total, gate_speed)
  an1 <- compare_RK_Groups(data)
  spaghetti_plot(data)
  print_Results(an1)
  return(an1)
}
width <- 200
height <- 50
pop_total <- 30 
gate_speed <- 1 

an1 <- main(width, height, pop_total, gate_speed)
