library(Ecdat)
library(ggplot2) 
library(ISLR)
library(GGally)
library(scatterplot3d)
library(censReg)
library(corrplot)
library(tidyr)
library(VGAM)
library(BBmisc)
library(gdata)
library(readxl)
library(missForest)
library(xlsx)
library(WriteXLS)
library(openxlsx)
library(dplyr)
library(car)

#get data
getwd()
dataframe <- read_excel("C:/Users/ShanoyaTaylor/Dropbox/Georgia Tech Courses/2019 - 4Spring/CSE6242/Project/Data/RemovedNA.xlsx", sheet = 5)
data <- data.matrix(dataframe[,7:38])
dataImp <- missForest(data, variablewise = TRUE)
dataDF <- dataframe[,1:6]
dataDF[,7:38] <- as.data.frame(dataImp$ximp)
write.xlsx(dataDF,"C:/Users/ShanoyaTaylor/Dropbox/Georgia Tech Courses/2019 - 4Spring/CSE6242/Project/Data/ImputedDataUpdate.xlsx", col.names = TRUE)
summary(dataDF)

#Testing imputation error
xVar <- read_excel("C:/Users/ShanoyaTaylor/Dropbox/Georgia Tech Courses/2019 - 4Spring/CSE6242/Project/Data/RemovedNA.xlsx", sheet = 1)
xtrue <- data.matrix(xVar[,7:38])
xmis <- prodNA(xtrue, noNA = 0.5)
ximp <- missForest(xmis, variablewise = TRUE)
r <- 1:32
#rsqexp <- expression(paste("R"^"2"))
#par(mfrow=c(3,4))
for(col in 23: ncol(xtrue))
{
  imperrreg <- lm(ximp$ximp[,col] ~ xtrue[,col])
  r[col]<- summary(imperrreg)$r.squared
  plot(xtrue[,col], ximp$ximp[,col], main = paste(colnames(xVar[,col+6])), xlab = paste("True "), ylab = paste("Imputed "))
  text(x = 0.5*(max(xtrue[,col])+min(xtrue[,col])), y = 0.97* max(ximp$ximp[,col]), paste("R sq =", sprintf("%.1f%%", 100*r[col])))
  mtext("Imputed versus True values for Process and Outcome measures with MAR data", outer = TRUE)
}


########Regression Models##############
trainindices <- sample(2477, 1730)
par(mfrow=c(3,3))

fullDF <- dataDF[,10:38]
fullDF <-select(fullDF, -Age0_14)
fulltrain <- fullDF[trainindices,]
fulltest <- fullDF[-trainindices,]
donDF <- dataDF[,29:38]
dontrain <- donDF[trainindices,]
dontest <- donDF[-trainindices,]

par(mfcol=c(4,7))
#Histogram
for(col in 1: ncol(fullDF)){
  hist(as.numeric(unlist(fullDF[,col])), main = paste("Distribution of \n", colnames(fullDF[,col])), xlab = colnames(fullDF[,col]))
  
}
###Mortality
#Process vs Mortality
mindonMort <- lm(Mortality ~ 1, data = dontrain)
maxdonMort <- lm(Mortality ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments, data = dontrain)
fwdDonMort <- step(mindonMort, scope = list(lower = mindonMort, upper = maxdonMort), direction = "both")
summary(fwdDonMort)
#predDonM = predict(fwdDonMort,newdata = dontest)
#plot(dontest$Mortality, predDonMort)
#Process + Socio vs Mortality
minfullMort <- lm(Mortality ~1, data = fulltrain)
maxfullMort <- lm(Mortality ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments + PCTCollegeDegree + PovertyRate + 
                    AvgHouseholdIncome + TotalPopulation + Males + Females + White + Black + Hispanic + Asian + Age15_29 + Age30_44 + 
                    Age45_59 + Age60_74 + Age75, data = fulltrain)
fwdFullMort <- step(minfullMort, scope = list(lower = minfullMort, upper = maxfullMort), direction = "both")
summary(fwdFullMort)
#predFullMort = predict(fwdFullMort,newdata = fulltest)
#plot(fulltest$Mortality, predFullMort)
#Process + Readmissions vs Mortality
mindonMortPR <- lm(Mortality ~1, data = dontrain)
maxdonMortPR <- lm(Mortality ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments + Readmissions, data = dontrain)
fwdDonMortPR <- step(mindonMortPR, scope = list(lower = mindonMortPR, upper = maxdonMortPR), direction = "both")
summary(fwdDonMortPR)
#predDonMort = predict(fwdDonMortPR,newdata = dontest)
#plot(dontest$Mortality, predDonMort)
#Process + Socio + Readimissions vs Mortality
minfullMortR <- lm(Mortality ~1, data = fulltrain)
maxfullMortR <- lm(Mortality ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments + PCTCollegeDegree + PovertyRate + 
                     AvgHouseholdIncome + TotalPopulation + Males + Females + White + Black + Hispanic + Asian + Age15_29 + Age30_44 + 
                     Age45_59 + Age60_74 + Age75 + Readmissions, data = fulltrain)
fwdFullMortR <- step(minfullMortR, scope = list(lower = minfullMortR, upper = maxfullMortR), direction = "both")
summary(fwdFullMortR)
#predFullMortR = predict(fwdFullMortR,newdata = fulltest)
#plot(fulltest$Mortality, predFullMortR)
anova(fwdDonMort, fwdFullMort)
anova(fwdDonMortPR, fwdFullMortR)

###Readmissions
#Process vs Readmissions
mindonRead <- lm(Readmissions ~1, data = dontrain)
maxdonRead <- lm(Readmissions ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments, data = dontrain)
fwdDonRead <- step(mindonRead, scope = list(lower = mindonRead, upper = maxdonRead), direction = "both")
summary(fwdDonRead)
#predDonRead = predict(fwdDonRead,newdata = dontest)
#plot(dontest$Readmissions, predDonRead)
#Process + Socio vs Readmissions
minfullRead <- lm(Readmissions ~ 1, data = fulltrain)
maxfullRead <- lm(Readmissions ~ HCAPS + StarRating + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments + PCTCollegeDegree + PovertyRate + 
                 AvgHouseholdIncome + TotalPopulation + Males + Females + White + Black + Hispanic + Asian + Age15_29 + Age30_44 + 
                 Age45_59 + Age60_74 + Age75, data = fulltrain)
fwdFullRead <- step(minfullRead, scope = list(lower = minfullRead, upper = maxfullRead), direction = "both")
summary(fwdFullRead)
#predFullRead = predict(fwdFullRead,newdata = fulltest)
#plot(fulltest$Readmissions, predFullRead)


###Rating 
#Process + vs Rating
mindonCens <- censReg(StarRating ~ HCAPS, data = dontrain, left = 1, right = 5)
maxdonCens <- censReg(StarRating ~ ., data = dontrain, left = 1, right = 5)
fwdDonCens = step(mindonCens, scope = list(lower = mindonCens, upper = maxdonCens), direction = "both")
mindonPOvsR <- lm(StarRating ~ 1, data = dontrain)
maxdonPOvsR <- lm(StarRating ~ HCAPS + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments, data = dontrain)
fwdDonRate <- step(mindonPOvsR, scope = list(lower = mindonPOvsR, upper = maxdonPOvsR), direction = "both")
summary(fwdDonRate)
#predDon = predict(fwdDonRate,newdata = dontest)
#plot(dontest$StarRating, predDon)
#Process + Socio vs Rating 
minfullCens <- censReg(StarRating ~ HCAPS, data = fulltrain, left = 1, right = 5)
maxfullCens <- censReg(StarRating ~ ., data = fulltrain, left = 1, right = 5)
fwdFullCens = step(minfullCens, scope = list(lower = minfullCens, upper = maxfullCens), direction = "both")
minfull <- lm(StarRating ~ 1, data = fulltrain)
maxfull <- lm(StarRating ~ HCAPS + MedicatComm + CleanQuiet + DoctorComm + Rncomm + DischargeInfo + Payments + PCTCollegeDegree + PovertyRate + 
                AvgHouseholdIncome + TotalPopulation + Males + Females + White + Black + Hispanic + Asian + Age15_29 + Age30_44 + 
                Age45_59 + Age60_74 + Age75, data = fulltrain)
fwdFull = step(minfull, scope = list(lower = minfull, upper = maxfull), direction = "both")
summary(fwdFull)
#predFull = predict(fwdFull,newdata = fulltest)
#plot(fulltest$StarRating, predFull)
anova(fwdDonRate, fwdFull)
#####################################
#######################################
#######################################
######################################