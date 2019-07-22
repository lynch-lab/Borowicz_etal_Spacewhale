from pathlib import Path
#import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.linear_model import RidgeClassifier as ridge_reg
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
import skimage.feature
from skimage.io import imread
from skimage.transform import resize


def load_image_files(container_path, dimension=(32, 32)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            #We extract image feature here, you can use different features from skimage.feature or even use them together by concatinate them into a single vector.
            flat_data.append(skimage.feature.hog(img_resized).flatten())
            #flat_data.append(skimage.feature.local_binary_pattern(img_resized, P=4, R=28).flatten())
     #       images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 DESCR=descr)

#load dataset and extract hog feature descriptor for each image
training_dataset = load_image_files("/gpfs/projects/LynchGroup/spacewhale/whale/tiled_air32/balanced_air32/train/")
testing_dataset = load_image_files("/gpfs/projects/LynchGroup/spacewhale/whale/new_pansharp/test")
X_train = training_dataset.data
y_train = training_dataset.target
X_test = testing_dataset.data
y_test = testing_dataset.target

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
#SVM classfier
#classifier = svm.SVC(C=10, class_weight='balanced')
#Ridge Regression classifier
classifier = ridge_reg(alpha = 1, class_weight='balanced')

#Prediction on training set.
classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_train)
y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(y_train, y_pred)))
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))

