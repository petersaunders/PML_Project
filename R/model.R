#Model building
require(caret)
require(randomForest)

setwd("D:/Documents/Other Courses/Practical Machine Learning/Project")

#Load data
trainDf = read.csv("./data/pml-training.csv")
testDf  = read.csv("./data/pml-testing.csv")

#Some columns contain a large number of NAs or empty fields
naCount      = apply(trainDf, 2, function(x)length(which(is.na(x))))
missingCount = apply(trainDf, 2, function(x)length(which(x == "")))
summaryColumns = sort(names(which(naCount > 0 | missingCount > 0)))

#Select the non-summary columns
windowColumns  = which(names(trainDf) %in% c("new_window", "num_window"))
summaryColumns = which(naCount > 0 | missingCount > 0)
trainSubDf = trainDf[, -c(windowColumns, summaryColumns)]

#Separate data into training and validation sets 
#and remove index, user and timestamp columns
set.seed(3485)
modelIdx = createDataPartition(trainSubDf$classe, p = 0.8, list = FALSE)

modelData       = trainSubDf[modelIdx, 6:58]
validationData  = trainSubDf[-modelIdx, 6:58]

#Now do k-fold cross-validation
K = 15
kfolds = createFolds(modelData$classe, k = K, list = TRUE, returnTrain = TRUE)
models = vector(length = K, mode = "list")

for (i in 1:K) {
    training = modelData[kfolds[[i]], ]  #make training
    testing  = modelData[-kfolds[[i]], ] #make testing
    
    #Make model and store it
    rfMod = randomForest(classe ~., data=training, ntree=500, norm.votes=FALSE)
    models[[i]] = rfMod
    
    #Evaluate on testing set to print accuracy
    preds = predict(rfMod, newdata=testing)
    
    cm = confusionMatrix(preds, testing$classe)
    cat("Accuracy[", i, "] = ", cm$overall['Accuracy'], "\n", sep="")
}

#Use all of the models to make predictions
validationPreds = lapply(models, function(x){predict(x, newdata=validationData)})
validationMatrix = matrix(unlist(validationPreds), ncol=nrow(validationData), nrow=K, byrow = TRUE)

#Use consensus vote of each of the K models to form prediction
vote <- function(x) {
    ux = unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}

finalPredict = apply(validationMatrix, MARGIN = 2, vote)

#Examine confusion matrix and statistics
cm = confusionMatrix(finalPredict, validationData$classe)

cat("Final validation accuracy =", cm$overall['Accuracy'], "\n")
print(cm$table)

#Do my predictions on the testing set
testSubDf = testDf[, -c(windowColumns, summaryColumns)]
testSubDf = testSubDf[, 6:58]

testSetPreds = lapply(models, function(x){predict(x, newdata=testSubDf)})
testSetMatrix = matrix(unlist(testSetPreds), ncol=nrow(testSubDf), nrow=K, byrow = TRUE)

finalTestPredict = apply(testSetMatrix, MARGIN = 2, vote)




