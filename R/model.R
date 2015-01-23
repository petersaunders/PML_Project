#Model building
setwd("D:/Documents/Other Courses/Practical Machine Learning/Project")

#Load packages
require(caret)
require(randomForest)

#Load training data
trainingDf = read.csv("./data/pml-training.csv")

#Find columns with NAs or empty strings
naCount      = apply(trainingDf, 2, function(x)length(which(is.na(x))))
missingCount = apply(trainingDf, 2, function(x)length(which(x == "")))
summaryColumns = sort(names(which(naCount > 0 | missingCount > 0)))

#Select the non-summary columns
windowColumns  = which(names(trainingDf) %in% c("new_window", "num_window"))
summaryColumns = which(naCount > 0 | missingCount > 0)
trainingDf = trainingDf[, -c(windowColumns, summaryColumns)]

#Remove index, user and timestamp columns
trainingDf = trainingDf[, 6:58]

#Separate data into training and validation sets
#and remove index, user and timestamp columns
set.seed(3485)
modelIdx = createDataPartition(trainingDf$classe, p = 0.8, list = FALSE)

modelData       = trainingDf[modelIdx, ]
validationData  = trainingDf[-modelIdx, ]

#Now do k-fold cross-validation
K = 15
kfolds = createFolds(modelData$classe, k = K, list = TRUE, returnTrain = TRUE)
models = vector(length = K, mode = "list")

for (i in 1:K) {
    kthTraining = modelData[kfolds[[i]], ]  #make training
    kthTesting  = modelData[-kfolds[[i]], ] #make testing
    
    #Make model and store it
    kthRF = randomForest(classe ~., data=kthTraining, ntree=500, norm.votes=FALSE)
    models[[i]] = kthRF
    
    #Evaluate on testing set to print accuracy
    kthPreds = predict(kthRF, newdata = kthTesting)
    
    cm = confusionMatrix(kthPreds, kthTesting$classe)
    cat("Accuracy[", i, "] = ", cm$overall['Accuracy'], "\n", sep="")
}

#Use all of the models to make predictions
validationPreds = lapply(models, function(x){predict(x, newdata = validationData)})
validationMatrix = matrix(unlist(validationPreds), ncol = nrow(validationData), nrow = K, byrow = TRUE)

#Use vote of each of the K models to form prediction
vote <- function(x) {
    ux = unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}

finalPredict = apply(validationMatrix, MARGIN = 2, vote)

#Examine confusion matrix and statistics
finalCm = confusionMatrix(finalPredict, validationData$classe)

cat("Final validation accuracy =", finalCm$overall['Accuracy'], "\n")
print(finalCm$table)


### Do my predictions on the testing set ###
testingDf  = read.csv("./data/pml-testing.csv")

testingDf = testingDf[, -c(windowColumns, summaryColumns)]
testingDf = testingDf[, 6:58]

testSetPreds = lapply(models, function(x){predict(x, newdata=testingDf)})
testSetMatrix = matrix(unlist(testSetPreds), ncol = nrow(testingDf), nrow = K, byrow = TRUE)

finalTestPredict = apply(testSetMatrix, MARGIN = 2, vote)


### Look at the most important variables in the models ###
require(AppliedPredictiveModeling)
require(ellipse)

topNImportant <- function(model, N=4) {
    imps     = importance(model)
    rankedVars = rownames(imps)[order(imps, decreasing=TRUE)]    
    topN = rankedVars[1:N]
    
    return(topN)
}

importantVars = sapply(models, topNImportant)
top4Overall   = apply(importantVars, 1, vote)

featurePlot(x = validationData[, top4Overall],
            y = validationData$classe,
            plot = "pairs",
            #visual args
            auto.key = list(columns = 5))




#Write out to file using supplied function
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

#Ensure character vector
answers = as.character(finalTestPredict)
pml_write_files(answers)






