#Reading the data and splitting it into training and cross #validation set
train <- read.csv("mnist_train.csv",header= FALSE)
splittingseed <- 100
set.seed(splittingseed )
selsize <- floor(0.70*nrow(train))
sel_ind <- sample(seq_len(nrow(train)),size=selsize)
trainset <- train[sel_ind,]
cvset <- train[-sel_ind,]
X <- trainset[,-1]
Y <- trainset[,1]
trainlabel <- trainset[,1]
cvlabel <- cvset[,1]

#Visualizing the training set images
attach(X)
par(mfrow=c(2,2),mai=c(0.1,0.1,0.1,0.1))
m <- matrix(unlist(X[3,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[12,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[48,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[212,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)


#Reducing Train and CV using PCA
Xreduced <- X/255
Xcov <- cov(Xreduced)
pcaX <- prcomp(Xcov)

# Creating a datatable to store and plot the 
# No of Principal Components vs Cumulative Variance Explained
vexplained <- as.data.frame(pcaX$sdev^2/sum(pcaX$sdev^2))
vexplained <- cbind(c(1:784),vexplained,cumsum(vexplained[,1]))
colnames(vexplained) <- c("No_of_Principal_Components","Individual_Variance_Explained","Cumulative_Variance_Explained")

#Plotting the curve using the datatable obtained
plot(vexplained$No_of_Principal_Components,vexplained$Cumulative_Variance_Explained, xlim = c(0,100),type='b',pch=16,xlab = "Principal Components",ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance Explained')

#Datatable to store the summary of the datatable obtained
vexplainedsummary <- vexplained[seq(0,100,5),]
vexplainedsummary

#Storing the vexplainedsummary datatable in png format for future reference.
library(gridExtra)
png("datatablevaraince explained.png",height = 800,width =1000)
p <-tableGrob(vexplainedsummary)
grid.arrange(p)
dev.off()



Xfinal <- as.matrix(Xreduced) %*% pcaX$x[,1:45]
cvreduced<- cvset[,-1]/255
cvfinal <- as.matrix(cvreduced) %*% pcaX$x[,1:45]

#Making training and cvset labels as factors
cvlabel <- as.factor(cvlabel)
trainlabel <- as.factor(trainlabel)

#Making a datatable to store errors for choosing the parameter C suing model selection
datatable_model_selection <- data.frame(c(0.1,0.5,1,10,20),c(NA,NA,NA,NA,NA),c(NA,NA,NA,NA,NA))
colnames(datatable_model_selection) <- c("C","Accuracytrain","AccuracyCV")

#Function to calculate accuracy for various C and store resultant 
#train and cv accuracy in datatable
calculate_accuracy<- function(variancefactor)
{
  require(e1071)
  svm.model <- svm(Xfinal,as.factor(trainlabel),cost = variancefactor)
  prediction <- predict(svm.model,Xfinal)
  table(prediction,trainlabel)
  correct <- prediction==trainlabel
  AccuracyTrain <- (sum(correct)/nrow(Xfinal))*100
  cat(sprintf("Accuracytrain:%f\n",AccuracyTrain))
  prediction2 <- predict(svm.model,cvfinal)
  table(prediction2,cvlabel)
  correct2<- prediction2==cvlabel
  AccuracyCV <- (sum(correct2)/nrow(cvfinal))*100
  cat(sprintf("Accuracycv:%f\n",AccuracyCV))
  return(c(AccuracyTrain,AccuracyCV))
}

#Applying the function to all the rows of datatable
for(j in 1:5)
{
  temp <-calculate_accuracy(datatable_model_selection[j,1])
  datatable_model_selection[j,2]<-temp[1]
  datatable_model_selection[j,3]<-temp[2]
}

#Adding columns for errors in datatable
datatable_model_selection$Trainset_Error <- (100-datatable_model_selection$Accuracytrain)
datatable_model_selection$CVset_Error <- (100-datatable_model_selection$AccuracyCV)
datatable_model_selection

#Storing the datatable in png format for future reference.
library(gridExtra)
png("datatable.png",height = 300,width = 4000)
p <-tableGrob(datatable_model_selection)
grid.arrange(p)
dev.off()

#Ploting the graph of C vs Errors
#Getting the subset of datatable that is to be plotted
plotdataframe <- datatable_no_of_nodes[,c(1,4,5)]

#melting the dataframe to feed to ggplot() function
library(reshape2)
meltedframe <- melt(plotdataframe,id="C")

#Applying ggplot() function
library(ggplot2)
finalplot<- ggplot(meltedframe,aes(C,value,color=variable))+geom_line()
+scale_colour_manual(values = c("red","blue"))
finalplot <- finalplot+xlab("C")+ylab("Error Value")
+ggtitle("Plot of C vs Error")
finalplot

#Hence the optimal value of C is 10
#Training SVM on total training set with C=10
#Applying PCA on total training set
totaltrain <- as.matrix(train[,-1])
totaltrainlabel <- as.matrix(train[,1])
totaltrainreduced <- totaltrain/255
totaltraincov <- cov(totaltrainreduced)
pcatotaltrain <- prcomp(totaltraincov)
totaltrainfinal <- as.matrix(totaltrain) %*% pcatotaltrain$x[,1:45]

#Applying SVM on total training set and calculating accuracy
totaltrainlabel <- as.factor(totaltrainlabel)
svm.model.final <- svm(totaltrainfinal,totaltrainlabel,cost = 10)
predictionfinaltrain <- predict(svm.model.final,totaltrainfinal)
correcttrainfinal <- predictionfinaltrain==totaltrainlabel
Accuracytrainfinal <- (sum(correcttrainfinal)/nrow(totaltrainfinal))*100
Accuracytrainfinal

#Load test to reduced and normalize it for predictions
test<- read.csv("mnist_test.csv",header=FALSE)
testlabel <- as.factor(test[,1])

#Applying PCA to test set
testreduced <- test[,-1]/255
testfinal <- as.matrix(testreduced) %*% pcatotaltrain$x[,1:45]

#Predicitng on the test set using trained SVM model
predictionfinaltest <- predict(svm.model.final,testfinal)

#Calculating test set accuracy
correcttestfinal <- predictionfinaltest==testlabel
Accuracytestfinal <- (sum(correcttestfinal)/nrow(testfinal))*100
#Final Train Accuracy =99.98
#Final Test Accuracy = 98.44




