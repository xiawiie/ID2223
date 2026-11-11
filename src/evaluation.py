import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

def per_class_accuracy(pred,truth,n=10):
    return [float((pred[truth==c]==c).mean()) for c in range(n)]

def report(pred,truth,names):
    return classification_report(truth,pred,target_names=names,digits=3),confusion_matrix(truth,pred)
