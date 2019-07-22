### Train/test 10-fold validation split

setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32')

water<-sample(list.files(path = "train//water"))
whale<-sample(list.files(path = "train//whale"))

length(water)/10 #is 1230.6 So we take 1230 for each one, and exclude some in the end
length(whale)/10 #is 23.9 so I guess 23

#Make a df of the subsets. 
#First set
water_new<-data.frame(filename=water[1:1230], set=1)
whale_new<-data.frame(filename=whale[1:23], set=1)

#Now add sets 2-10
for(i in 2:10){
  water_new<-rbind(water_new, data.frame(filename=water[((i-1)*1230+1):(i*1230)], set=i))    
  whale_new<-rbind(whale_new, data.frame(filename=whale[((i-1)*23+1):(i*23)], set=i))
}

#setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32\\10_fold')

## OK so we have a list of all the files, divided into 10.
## Now we need to combine these into groups. So we'll have set 1 = train on 2:10, test on 1
## Then set2 = train on 1, 3:10, test on 2
## So we can say put all the water that is not 1 in new dir
## and put 1 in dir. Then same with whale.
## file.copy(listoffiles, new_dir)

# Make a list of the dirs


for(i in 1:10){
  # Training set
  setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32\\train\\water')
  base_dir<-"..//..//10_fold//fold_"
  to_dir<-c(paste(base_dir,i,"//", sep=""))
  # copy all the water files that aren't i to training
  file.copy(water_new[which(water_new[,2] != i),1], paste(to_dir,"train//water", sep=""))
  
  # now copy all the files that are i to testing
  file.copy(water_new[which(water_new[,2] == i),1], paste(to_dir,"test//water", sep=""))
  
  # Now whales
  setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32\\train\\whale')
  file.copy(whale_new[which(whale_new[,2] != i),1], paste(to_dir,"train//whale", sep=""))
  file.copy(whale_new[which(whale_new[,2] == i),1], paste(to_dir,"test//whale", sep=""))
}
file.copy(water_new[which(water_new[,1] != i),1], "10_fold//fold_1//train//water")


### Now set up a smaller, more balanced training set.
### So we want all the whales, and an equal number of waters

bal_set_water<-sample(water, length(whale), replace=FALSE)

setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32\\train\\water')
file.copy(bal_set_water, "..//..//..//balanced_air32//train//water")
setwd('C:\\Users\\Starship\\Downloads\\temp\\tiled_air32\\train\\whale')
file.copy(whale, "..//..//..//balanced_air32//train//whale")


