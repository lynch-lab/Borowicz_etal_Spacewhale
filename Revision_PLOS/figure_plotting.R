library(tidyverse) #for ggplot
library(gridExtra) #to align plots
## Colorblind color palette, plus extras
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00","#F0E442", "#000000", "#54278f", "grey62" )
## Layout linetypes for plotting
plottinglines<- c("solid","dashed", "dotted","dotdash","F1")

## Load in data
setwd('C:\\Users\\Starship\\Desktop\\GitHub\\spacewhale\\Revision_PLOS')
setwd('C:\\Users\\Alex\\Documents\\GitHub\\spacewhale\\Revision_PLOS')
#res: A CSV of the acc and loss of all models. Cols: epoch, acc, loss, LR, model (e.g. resnet-152)
res<-read.csv('train_model_results.csv') 
res$LR<-as.factor(res$LR) #Make LR a factor for plotting

##res2: the results of testing models. cols: prec, recall, F1, model, LR, epoch
res2<-read.csv('test_model_results.csv')
foldres<-read.csv('Fold10_training_results.csv')
foldres2<-read.csv('test_fold_results.csv')
confu_dat<-read.csv('confusion_data.csv')
tt_data<-read.csv('res152_traintest_curve.csv')
tt_data$mode<-as.factor(tt_data$mode)
### Data wrangling ###

res$LR<-as.factor(res$LR)
res$model<-factor(res$model, levels=c("resnet-18","resnet-32","resnet-152","densenet"))
res2$LR<-as.factor(res2$LR)
res2$model<-factor(res2$model, levels=c("resnet-18","resnet-32","resnet-152","densenet"))
foldres$model<-factor(foldres$model, levels = c("Fold 1",  "Fold 2" , "Fold 3",  "Fold 4" , "Fold 5" , "Fold 6" , "Fold 7"  ,"Fold 8" , "Fold 9",  "Fold 10"))

foldlist<-c("Fold 1",  "Fold 2" , "Fold 3",  "Fold 4" , "Fold 5" , "Fold 6" , "Fold 7"  ,"Fold 8" , "Fold 9",  "Fold 10")
foldres$model<-factor(foldres$model, levels = foldlist)
foldres2$model<-factor(foldres2$model, levels = foldlist)
p<-ggplot(data=res, aes(x=seq(1,length(res$tp))))+
  #geom_point(aes(y=acc, x=seq(1,length(res$tp))))+
  #geom_point(aes(y=tn), color='blue')+
  #geom_point(aes(y=tp), color='red')+
  geom_line(aes(y=acc), color='red')+
  geom_line(aes(y=loss), color='blue')+
  theme_minimal()+
  labs(x='Epoch', y='')


p

### This one plots each model train as a line.
### There's a diff color for each LR, diff line type for each architect

#####################################
### Plotting the LR experiment ######
### Show each LR for Res18 ##########
### "Effect of Learning rate on #####
### Training accuracy" ##############
#####################################

lr_p<-ggplot(data=res[which(res$model == "resnet-152"),], aes(x=epoch))+
  geom_line(aes(y=Acc, color=LR, linetype=model), size=1, alpha=0.6)+
  theme_minimal()+
  scale_colour_manual(values=cbPalette)+
  scale_linetype_manual(values=plottinglines)+
  theme(legend.key.width=unit(1,"cm"))

lr_p
#####################################
### Model training results ##########
### We're plotting accuracy and loss#
### Each LR is a color. Can add     #
### linetype for model type         #
#####################################
acc_p<-ggplot(data=res, aes(x=epoch))+
  #geom_line(aes(y=Acc, color=LR))+
  #geom_point(aes(y=Acc, color=LR, shape=model))+
  #geom_line(aes(y=Acc, color=LR, linetype=model), size=1)+
  #geom_line(data=res[which(res$model == 'densenet'),], aes(y=Acc,x=epoch, color=LR), size=1.5)+
  #geom_line(data=res[which(res$model != 'ResneXt'),], aes(y=Acc,x=epoch, color=LR, linetype=model), size=1)+
  geom_line(data=res[which(res$model == 'resnet-152'),], aes(y=Acc, x=epoch, color=LR), size=1)+#, alpha=0.6)+
  theme_minimal()+
  scale_colour_manual(values=cbPalette)+
  scale_linetype_manual(values=plottinglines)+
  theme(legend.key.width=unit(1,"cm"))+
  labs(x="Epoch", y="Accuracy")+
  lims(y=c(0.5,1))
acc_p

los_p<-ggplot(data=res, aes(x=epoch))+
  #geom_line(aes(y=Acc, color=LR))+
  #geom_point(aes(y=Acc, color=LR, shape=model))+
  geom_line(aes(y=Loss, color=LR, linetype=model), size=1)+
  theme_minimal()+
  scale_colour_manual(values=cbPalette)
los_p


##################################################
### Training Testing Curve - we compare acc ######
### Across the epochs to demonstrate overfit #####
##################################################
##Full Acc - train/test
##Water Acc - train test
## Whale acc - train/test
## Train /test color
## Full - solid
## Water/whale - dot/dash

tt_curve<-ggplot(data=tt_data, aes(x=epoch))+
  geom_line(aes(y=acc, color=mode), size=1)+
  geom_line(aes(y=wat_acc, color=mode), linetype=4, size=1)+
  geom_line(aes(y=whale_acc, color=mode), linetype=3, size=1)+
  theme_minimal()+
  scale_colour_manual(values=cbPalette)+
  labs(x="Epoch",y="Accuracy")+
  lims(y=c(0.5,1))
  
tt_curve

# Figure 3 - put acc plot and train/test curve together
# In Illustrator, font, size, labels, make acc bars transpar
grid.arrange(acc_p, tt_curve, ncol=2)

##################################################
### Now plotting the test results for each model #
##################################################
#### Figure 4 - Model Performance ################
testres<-ggplot(data=res2, aes(x=prec, y=recall, color=LR, shape=model))+
  #geom_point()+
  theme_minimal()+
  # We jitter the points because some are overlapping
  #geom_jitter(aes(color=LR, shape=model), width=0.0007, size=3)+
  geom_jitter(aes(color=LR, shape=model), width=0.0015, size=3)+
  scale_colour_manual(values=cbPalette) #Use the colorblind palette
testres

###############################################
### 10-fold Training Results ##################
###############################################

foldplot<-ggplot(data=foldres, aes(x=epoch))+
  # Plotting both acc and loss on same plot
  geom_line(aes(y=Acc, color=model), size=0.5)+
  geom_line(aes(y=Loss, color=model), linetype='dotdash', size=0.5)+
  theme_minimal()+
  scale_colour_manual(values=cbPalette)+
  labs(x="Training Epoch", y="Accuracy (solid) and Loss (dashed)")+
  ggtitle("Ten-fold model training performance")+
  ylim(c(0,1))
foldplot

###############################################
### 10-fold Test results ######################
###############################################

foldtest<-ggplot(data=foldres2, aes(x=prec, y=recall, color=model))+
  #geom_point(size=3)+
  theme_minimal()+
  # Jittering again to separate overlapping pts
  geom_jitter(aes(), width=0.0002, height=0.0002, size=3)+
  geom_jitter(aes(), width=0.0004, height=0.0003, size=4)+
  scale_colour_manual(values=cbPalette)+
  labs(x="Precision", y="Recall")
foldtest

ggplot(data=foldres2)+
  geom_histogram(aes(recall))
########### Confusion matrix

#############################################################################
## Making confusion matrices for the different model runs ###################
## c() for each is tp,tn,fp,fn where tp is a true water, tn is a true whale #
## Just change the model and LR to pull out a new conf mat ##################

## Just change the model and LR to pull out a new conf mat ##################

confu_dat$lr<-as.factor(confu_dat$lr)
confu_dat$lab<-as.factor(confu_dat$lab)
confu_dat$pred<-as.factor(confu_dat$pred)

ggplot(data =  confu_dat[which(confu_dat$mod=='resnet-152' & confu_dat$lr=='0.2'),], aes(x = lab, y = pred)) +
  geom_tile(aes(fill = val), colour="black") +
  geom_text(aes(label = sprintf("%1.0f", val)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "white") +
  theme_bw() + theme(legend.position = "none")+
  xlab("True Class")+
  ylab("Predicted Class")+
  ggtitle(confu_dat$mod, confu_dat$lr)
  #ggtitle(confu_dat$mod, confu_dat$lr)

den001<-c(1281,31,1,109)
den01<-c(1015,28,4,375)
den1<-c(714,29,3,676)
r18_0009<-c(1296,32,0,94)
r18_001<-c(1257,32,0,3)
r18_01<-c(1129,26,6,261)
r32_01<-c(1295, 19, 13, 95)
