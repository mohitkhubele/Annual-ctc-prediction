
## Loading the required libraries 
library(ipred)
library(ROSE)
library(ada)
library(rpart.plot)
library(rpart)
library(randomForest)
library(C50)
library(factoextra)
library(xgboost)
library(glmnet)
library(mice)
library(dplyr)
library(ROCR)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(e1071)
library(corrplot)
library(dplyr)
library(purrr)

rm(list = ls(all=TRUE))
setwd("C:/Users/Lenovo/Desktop/mith")
#read the data
train<-read.csv("train.csv",header = T,sep = ",")
test<-read.csv("test.csv")
train_label<-train$Pay_in_INR
train$Pay_in_INR<-NULL
train<-rbind(train,test)
rm(test)
#Drop the duplicate records.

train<-train[!duplicated(train),]
test<-test[!duplicated(test),]
#there are no duplicates.
#check the structure and summary of data
summary(train)
str(train)

#remove unnecessary columns 
length(unique(train$Candidate.ID))
train$Candidate.ID<-NULL
train$Date.Of.Birth<-NULL
train$CollegeCode<-NULL
train$CityCode<-NULL

#for column School.Board.in.Tenth
train$School.Board.in.Tenth <- na_if(train$School.Board.in.Tenth,0)
sum(is.na(train$School.Board.in.Tenth))#7283 missing values

#for column Board in twelth
train$Board.in.Twelth <- na_if(train$Board.in.Twelth,0)
sum(is.na(train$Board.in.Twelth))#7296 missing values

#for column Score in Domain
train$Score.in.Domain <- na_if(train$Score.in.Domain,-1)
sum(is.na(train$Score.in.Domain))#1002

# convertion of apropriate data types
train$CollegeTier<-as.factor(train$CollegeTier)
train$CityTier<-as.factor(train$CityTier)
str(train)

#handling missing values with central imputation

pretrain<-centralImputation(train)
sum(is.na(pretrain))

summary(pretrain)
colnames(pretrain)


#find out relationship between dependent and independent variables
par(mfrow = c(2,2))
plot(pretrain$Gender, pretrain$Pay_in_INR, xlab = "gender", ylab = "pay_in_INR", main = "gender vs pay")
hist(pretrain$Score.in.Tenth, xlab="Score.in.Tenth")

table(pretrain$Discipline)

Discipline.bins <- function(x){
  if(x == "computer application" | x == "computer science & engineering"|x=="electronics and electrical engineering"|x=="mechanical engineering")
    dis <- '1'    
  else if(x == "computer engineering" | x=="electronics and communication engineering" | x=="information technology")
    dis <- '2'    
  else if(x == "civil engineering" | x=="electrical engineering" | x=="electronics & instrumentation eng" | x=="industrial & production engineering" | x=="instrumentation and control engineering")
    dis <- '3'     
  else 
    dis <- '4'    
  
  return(dis)
}

Discipline <- data.frame("Discipline.bins"=sapply(pretrain$Discipline,Discipline.bins))
table(Discipline)
pretrain$Discipline<-NULL

#binning Board of tenth
table(pretrain$School.Board.in.Tenth)

Tenth.bins <- function(x){
  if(x == "cbse" | x=="central board of secondary education")
    board <- '1'    
  else if(x == "state board" | x=="state")
    board <- '2'    
  else if(x == "icse")
    board <-"3"
  else if(x=="up"|x=="up board")
    board <- '4' 
  else if(x=="ssc"| x=="sslc")
    board <- '4'
  else 
    board <- '5'    
  
  return(board)
}

Tenth_Board <- data.frame("Tenth.bins"=sapply(pretrain$School.Board.in.Tenth,Tenth.bins))
table(Tenth_Board)
pretrain$School.Board.in.Tenth<-NULL

#binning board of inter

inter.bins <- function(x){
  if(x == "cbse" | x=="central board of secondary education")
    board <- '1'    
  else if(x == "state board" | x=="state")
    board <- '2'    
  else if(x == "icse" | x=="isc")
    board <-"3"
  else if(x=="up"|x=="up board")
    board <- '4' 
  else if(x=="board of intermediate"| x=="board of intermediate education" | x=="board of intermediate education ap")
    board <- '5'
  else 
    board <- '6'    
  
  return(board)
}

inter_Board <- data.frame("inter.bins"=sapply(pretrain$Board.in.Twelth,inter.bins))
table(inter_Board)
pretrain$Board.in.Twelth<-NULL

#bind all coulmns with binning values
pretrain<-cbind(pretrain,Discipline,Tenth_Board,inter_Board)
rm(Discipline)
rm(Tenth_Board)
rm(inter_Board)

str(pretrain)


# find out graduation years
twelth<-pretrain$Year.Of.Twelth.Completion
graduation<-pretrain$Year.of.Graduation.Completion
dif<-data.frame(cbind(twelth,graduation))
dif$years<- "NA"
for(i in nrow(dif))
  dif$years<-dif$graduation[i]-dif$twelth

dif$twelth<-NULL
dif$graduation<-NULL

pretrain$Year.Of.Twelth.Completion<-NULL
pretrain$Year.of.Graduation.Completion<-NULL
pretrain<-cbind(pretrain,dif)
pretrain$years<-as.factor(pretrain$years)

str(pretrain)
pretrain$State<-NULL

#splitting train and test data
train<-pretrain[1:26415,]
test<-pretrain[26416:37290,]
pretrain<-data.frame(cbind(train,train_label))

#########model building #############
# multiple linear regression
LinReg1<- lm(train_label~ ., data=pretrain)
summary(LinReg1)
#here Score.in.Tenth GPA.Score.in.Graduation , Score.in.Domain ,Score.in.ElectronicsAndSemicon ,Score.in.CivilEngg are not significant.
head(predict(LinReg1))

#Review the residual plots
par(mfrow=c(2,2))
plot(LinReg1)
plot(LinReg1,which=4)
#cooks distance for 3 points are high 19692,21382,26293

# Error metrics evaluation on train data and test data
#Error verification on train data
regr.eval(pretrain$train_label, LinReg1$fitted.values)

vif(LinReg1)
# Stepwise Regression
Step3 <- stepAIC(LinReg1, direction="both")
LinReg2<- lm(train_label ~ Gender + Score.in.Twelth + CollegeTier + Graduation + 
               CityTier + Score.in.English.language + Score.in.Logical.skill + 
               Score.in.Quantitative.ability + Score.in.Domain + Score.in.ComputerProgramming + 
               Score.in.ComputerScience + Score.in.MechanicalEngg + Score.in.ElectricalEngg + 
               Score.in.TelecomEngg + Score.in.conscientiousness + Score.in.agreeableness + 
               Score.in.extraversion + Score.in.nueroticism + Score.in.openess_to_experience + 
               Discipline.bins + Tenth.bins + inter.bins + years , data=pretrain)
summary(LinReg2)# all are significant except intercept
vif(LinReg2)#their is no multi colinearity

########### PCA########
pre_std_train<-preProcess(pretrain[,setdiff(colnames(pretrain),"train_label")])
train_scale<-predict(pre_std_train,pretrain[,setdiff(colnames(pretrain),"train_label")])

dummies <- dummyVars(train_label~., data = pretrain)
x.train=predict(dummies, newdata = pretrain)
y.train=pretrain$train_label
Train<-data.frame(x.train)
prcomp_train <- princomp(Train)
prcomp_train
plot(prcomp_train)

#12 components cover 99% data
train_data<-prcomp_train$scores
summary(prcomp_train)

##subset the components identified from train_data
compressed<-train_data[,1:11]

#Bind Target train$Target with the output

final_train<-data.frame(cbind(y.train,compressed))
# build model on PCA
LinReg3<-lm(y.train~.,data = final_train)
summary(LinReg3)
#Error verification on train data
regr.eval(final_train$y.train, LinReg3$fitted.values)

#######################cart########################
#Build a regression model using rpart
DT_rpart_Reg<-rpart(train_label~.,data=pretrain,method="anova")

DT_rpart_Reg<-rpart(train_label~.,data=pretrain,method="anova",control = rpart.control(cp = 0.0012))
printcp(DT_rpart_Reg)
print(DT_rpart_Reg)



predCartTrain=predict(DT_rpart_Reg, newdata=pretrain, type="vector")
predCartTest=predict(DT_rpart_Reg, newdata=test, type="vector")

regr.eval(pretrain[,"train_label"], predCartTrain)

#####################gbm###################
test$test_label<-"NA"
# Bulding gbm using caret package in R
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init()


# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = pretrain, destination_frame = "train.hex")


# Prepare the parameters for the for H2O gbm grid search
ntrees_opt <- c(5, 10, 15, 20, 30)
maxdepth_opt <- c(2, 3, 4, 5)
learnrate_opt <- c(0.01, 0.05, 0.1, 0.15 ,0.2, 0.25)
hyper_parameters <- list(ntrees = ntrees_opt, 
                         max_depth = maxdepth_opt, 
                         learn_rate = learnrate_opt)

# Build H2O GBM with grid search
grid_GBM <- h2o.grid(algorithm = "gbm", grid_id = "grid_GBM.hex",
                     hyper_params = hyper_parameters, 
                     y = "train_label", x = setdiff(names(train.hex), "train_label"),
                     training_frame = train.hex)

# Remove unused R objects
rm(ntrees_opt, maxdepth_opt, learnrate_opt, hyper_parameters)

# Get grid summary
summary(grid_GBM)

# Fetch GBM grid models
grid_GBM_models <- lapply(grid_GBM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_rmse = h2o.rmse(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_rmse = h2o.rmse(temp_model)
    if(best_model_rmse < temp_model_rmse)
    {
      best_model = temp_model
      best_model_rmse = temp_model_rmse
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GBM_model = find_Best_Model(grid_GBM_models)

rm(grid_GBM_models)
best_GBM_model_rmse = h2o.rmse(best_GBM_model)#329281.4

# Examine the performance of the best model
best_GBM_model

# View the specified parameters of the best model
best_GBM_model@parameters

# Important Variables.
varImp_GBM <- h2o.varimp(best_GBM_model)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test, destination_frame = "test.hex")

# Predict on same test data set
predict.hex = h2o.predict(best_GBM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "test_label")])

data_GBM = h2o.cbind(test.hex[,"test_label"], predict.hex)

# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)

#Shutdown H2O
h2o.shutdown(F)
test$test_label<-NULL
#######################Random Forest#######################
model_rf <- randomForest( train_label~ ., data= pretrain)

#its taking so much of time so shifting to xgboost

###########xgboost#######
#create matrix - One-Hot Encoding for factor variable
pretrain<-cbind(train,train_label)
trainm<- sparse.model.matrix(train_label ~ ., data = pretrain)
train_label<- pretrain[,"train_label"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

#for test data
test$Pay_in_INR<-"NA"
testm<- sparse.model.matrix(Pay_in_INR ~ ., data = test)
test_label<- test[,"Pay_in_INR"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)
#parameters
xgb_params <- list("objective" = "reg:linear", "eval_metric" = "rmse")
watchlist<- list(train = train_matrix, test = test_matrix)
####model building
bst_model<-xgb.train(params = xgb_params,data = train_matrix,nround = 10000, watchlist = watchlist)

###error plot
e<-data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = "blue")
lines(e$iter, e$test_mlogloss, col = "red")

# Feature importance
imp<- xgb.importance(colnames(train_matrix),model = bst_model)
print(imp)
xgb.plot.importance(imp)

# prediction & confusion matrix - test data
p<-predict(bst_model, newdata = test_matrix)
p1<-data.frame(p)

##again build with tuning parameters
bst_model2<-xgb.train(params = xgb_params,data = train_matrix,nround = 3000, watchlist = watchlist,eta = 0.5)
# Feature importance
imp<- xgb.importance(colnames(train_matrix),model = bst_model)
print(imp)
xgb.plot.importance(imp)

# prediction & confusion matrix - test data
q<-predict(bst_model, newdata = test_matrix)
q1<-data.frame(q)

##so best model is xgboost

###select features from xgboost den build mulitiple model on that
LinReg_feature<- lm(train_label ~ Score.in.Quantitative.ability  + Score.in.conscientiousness  + Score.in.Tenth  + Score.in.Twelth  + 
                      Score.in.ComputerProgramming  + Score.in.Logical.skill  + Score.in.openess_to_experience  + 
                      GPA.Score.in.Graduation  + Score.in.English.language  + Score.in.agreeableness  + 
                      Score.in.nueroticism  + Score.in.extraversion  + Score.in.Domain  + 
                      Score.in.ComputerScience  + Score.in.ElectronicsAndSemicon  + CollegeTier, data=pretrain)
summary(LinReg_feature)
vif(LinReg_feature)
regr.eval(pretrain$train_label, LinReg_feature$fitted.values)
pred<-predict(LinReg_feature,test)
pred1<-data.frame(pred)
write.csv(pred1,"pred1.csv", row.names=F)
write.csv(p1,"pred2.csv")
write.csv(q1,"pred3.csv")
