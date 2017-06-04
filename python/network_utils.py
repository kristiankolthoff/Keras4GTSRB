import numpy as np
import keras.utils as utils
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import f1_score

def preprocess(label, classes):
    palette = sorted(list(set(label)))
    index = np.digitize(label.ravel(), palette, right=True)
    key = np.array(list(range(0,classes)))
    label = key[index].reshape(label.shape)
    label = utils.to_categorical(label, classes)
    return label

def transformLabel(label, classes):
    palette = sorted(list(set(label)))
    index = np.digitize(label.ravel(), palette, right=True)
    key = np.array(list(range(0,classes)))
    label = key[index].reshape(label.shape)
    return label

def fromCategorical(label):
    return np.argmax(label,axis=1)

def computeClassWeights(label, fractionReducer):
    class_freq = Counter(label)
    max_weight = class_freq[max(class_freq)]
    class_weights = {key: ((max_weight/value)/fractionReducer) 
                    for key, value in class_freq.most_common()}
    return class_weights

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def get_f1_per_group(labels, prediction, average="micro"):
    f1_score_speed_limit = f1_score(labels, prediction, 
                                    average=average, labels=list(range(0,8)))
    f1_score_prohib = f1_score(labels, prediction, 
                                    average=average, labels=list(range(8,12)))
    f1_score_derest = f1_score(labels, prediction, 
                                    average=average, labels=list(range(12,16)))
    f1_score_mand = f1_score(labels, prediction, 
                                    average=average, labels=list(range(16,24)))
    f1_score_danger = f1_score(labels, prediction, 
                                    average=average, labels=list(range(24,39)))
    f1_score_unique = f1_score(labels, prediction, 
                                    average=average, labels=list(range(39,43)))
    return {"speed":f1_score_speed_limit,"prohib":f1_score_prohib,
            "derest": f1_score_derest, "mand":f1_score_mand,
            "danger":f1_score_danger, "unique":f1_score_unique}