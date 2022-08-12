library("lmerTest")
library("multcomp")
library("emmeans")
library("readxl")
options(max.print=1000000)

NEW_DATA$Injury = as.factor(NEW_DATA$Injury)
NEW_DATA$Time = as.factor(NEW_DATA$Time)
NEW_DATA$Mouse = as.factor(NEW_DATA$Mouse)

results = lmer(Amplitude ~ Injury * Time + (1|Mouse), data = NEW_DATA)
anova(results, type = "I")
#Only Time a significant effect for post hoc analysis
#            Sum Sq Mean Sq NumDF   DenDF F value   Pr(>F)    
# Injury       5.960  5.9604     1  22.058  3.6775  0.06819 .  
# Time        68.781  7.6423     9 196.235  4.7152  1.11e-05 ***
# Injury:Time 26.632  2.9591     9 196.250  1.8257  0.06568 .

summary(glht(results, emm(pairwise ~ Time)), test = adjusted("holm"))
#                       Estimate  Std. Error t value Pr(>|t|)
# (-0.014) - 0.014 == 0 -0.110916   0.368795  -0.301 1.000000    
# (-0.014) - 0.042 == 0 -0.154599   0.372506  -0.415 1.000000    
# (-0.014) - 1 == 0     -1.437111   0.368795  -3.897 0.005483 ** 
# (-0.014) - 3 == 0     -1.710558   0.374047  -4.573 0.000383 ***
# (-0.014) - 7 == 0     -0.836336   0.368795  -2.268 0.806358    
# (-0.014) - 14 == 0    -0.722217   0.368795  -1.958 1.000000    


results = lmer(Area ~ Injury * Time + (1|Mouse), data = NEW_DATA)
anova(results, type = "I")
#Significant Injury and Time effects for post hoc analysis
#            Sum Sq Mean Sq NumDF   DenDF F value    Pr(>F)    
#Injury      22.638  22.639     1  23.905 11.1831  0.002714 ** 
#Time        94.166  10.463     9 183.588  5.1686 2.979e-06 ***
#Injury:Time 97.022  10.780     9 173.850  6.8911 1.827e-08 ***

summary(glht(results, emm(pairwise ~ Injury * Time)), test = adjusted("holm"))



# (Inj -0.014) - (Sham -0.014) == 0  0.1448619  0.5518699   0.262 1.000000    
# (Inj -0.014) - Inj 0.014 == 0     -3.3512122  0.4905838  -6.831 2.59e-08 ***
# (Inj -0.014) - Sham 0.014 == 0     0.2074869  0.5518699   0.376 1.000000    
# (Inj -0.014) - Inj 0.042 == 0     -2.7067728  0.5015918  -5.396 3.85e-05 ***
# (Inj -0.014) - Sham 0.042 == 0     0.0375944  0.5656416   0.066 1.000000    
# (Inj -0.014) - Inj 1 == 0         -0.5857952  0.5286767  -1.108 1.000000    
# (Inj -0.014) - Sham 1 == 0        -0.2937747  0.6258558  -0.469 1.000000    
# (Inj -0.014) - Inj 3 == 0         -1.9893965  0.5141327  -3.869 0.023923 *  
# (Inj -0.014) - Sham 3 == 0         0.1146315  0.5656416   0.203 1.000000    
# (Inj -0.014) - Inj 7 == 0         -0.1324642  0.5015918  -0.264 1.000000    
# (Inj -0.014) - Sham 7 == 0        -0.3609143  0.5818729  -0.620 1.000000    
# (Inj -0.014) - Inj 14 == 0        -0.0159021  0.5139088  -0.031 1.000000    
# (Inj -0.014) - Sham 14 == 0        0.0767405  0.5817444   0.132 1.000000

# Inj 0.014 - Sham 0.014 == 0        3.5586991  0.5518699   6.448 2.03e-07 ***
# Inj 0.042 - Sham 0.042 == 0        2.7443672  0.5752149   4.771 0.000632 ***
# Inj 1 - Sham 1 == 0                0.2920205  0.6561418   0.445 1.000000 
# Inj 3 - Sham 3 == 0                2.1040280  0.5861829   3.589 0.064288 . 
# Inj 7 - Sham 7 == 0               -0.2284501  0.5911835  -0.386 1.000000   
# Inj 14 - Sham 14 == 0              0.0926426  0.6015450   0.154 1.000000 
                              