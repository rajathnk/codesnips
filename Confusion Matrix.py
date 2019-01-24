# To calculate confusion matrix in case of classifictaion problems
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_ground_truth, y_model_prediction)
    pos_class_acc = CM[1,1]/(CM[1,0]+CM[1,1])
    neg_class_acc = CM[0,0]/(CM[0,0]+CM[0,1])
    print("Accuracy of positive class:",pos_class_acc)
    print("Accuracy of negative class:",neg_class_acc)
    print('Balanced accuracy', (pos_class_acc+neg_class_acc)/2)
