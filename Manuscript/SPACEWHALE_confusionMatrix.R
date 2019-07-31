'''
SPACEWHALE PROJECT
Author: A. Borowicz (aborowicz@coa.edu)
https://github.com/aborowicz/spacewhale

This script takes outputs from the testing stage: 3 messy CSVs where
the true labels (whale or water as 0,1), predictions(same), and image
file paths each have their own CSV. But, since the model takes them 
as batches, each CSV has 10 columns which are meaningless.
This can be cleaned into one dataframe, where the cols are labels, 
predictions, names. Then a confusion matrix is needed.
'''

library(tidyverse)
library(ggplot2)
library(reshape2)
library(scales)
library(glue)
# Read in the 3 CSVs
setwd('C:\\Users\\Starship\\Google Drive\\Projects\\Space Whales')
preds<-    read.csv('predicted.csv',   header=TRUE)
labs<-     read.csv('labeled.csv',     header=TRUE)
filenames<-read.csv('file_output.csv', header=TRUE, stringsAsFactors = FALSE)


predvec<-c()
labvec<-c()
filevec<-c()

#Iterate through all the csvs and straighten them out
for(i in 2:length(preds[,1])){
  for(j in 1:length(preds[i,])){
    if (is.na(preds[i,j])==T)
      next
    
      predvec<-c(predvec, preds[i,j])
      labvec <-c(labvec,  labs[i,j])
      filevec<-c(filevec, as.vector(filenames[i,j]))
  }
}

filevec<-regmatches(filevec, regexpr("[w].*[.png]", filevec))
dat<-tibble('image'=filevec, 'label'=as.factor(labvec), 'prediction'=as.factor(predvec), 'result'=NA)

### OK We have our data in a nice tibble. Now we need to pull out 4 values for the confusion matrix
### True Pos, True Neg, False pos, false neg. Water is positive, whale negative). I'm sure there's a
### slicker way to do this than a gazillion if elses

for(i in 1:length(predvec)){
  if (sum(dat$label[i], dat$prediction[i])==0)
    dat$result[i]<-'TN'
  else 
    if (sum(dat$label[i], dat$prediction[i])==2)
      dat$result[i]<-'TP'
    else
      if (dat$label[i]==1)
        dat$result[i]<-'FN'
      else
        if (dat$prediction[i]==1)
          dat$result[i]<-'FP'
}

###Things for plotting
# Calculate # of Trues and falses for the whole set
results<-c()
results<-c(results,length(which(dat$label==0 & dat$prediction==0))) #True Neg
results<-c(results,length(which(dat$label==1 & dat$prediction==1))) #True Pos
results<-c(results,length(which(dat$label==1 & dat$prediction==0))) #False Neg
results<-c(results,length(which(dat$label==0 & dat$prediction==1))) #False Pos

# 

TrueClass<-factor(c('Water','Water','Whale','Whale'))
PredClass<-factor(c('Water','Whale','Water','Whale'))
df<-data.frame(TrueClass,PredClass,results)


### Confusion matrix. Need a new one for each model run.
ggplot(data =  df, mapping = aes(x = TrueClass, y = PredClass)) +
  geom_tile(aes(fill = results), colour="black") +
  geom_text(aes(label = sprintf("%1.0f", results)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "white") +
  theme_bw() + theme(legend.position = "none")+
  xlab("True Class")+
  ylab("Predicted Class")



