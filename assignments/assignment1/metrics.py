import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = len([1 for i in range(len(prediction)) if (ground_truth[i] == prediction[i] == True)])
    tn = len([1 for i in range(len(prediction)) if (ground_truth[i] == prediction[i] == False)])
    fp = len([True for i in range(len(prediction)) if (prediction[i] == True and ground_truth[i] == False)])
    fn = len([True for i in range(len(prediction)) if (prediction[i] == False and ground_truth[i] == True)])
    #print(tp, tn, fp, fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fn + fp) 
#     num_samples = len(prediction)
    
#     tp = np.sum([ground_truth[i] == prediction[i] == True for i in range(num_samples)])
#     fp = np.sum([True for i in range(num_samples) \
#                 if ground_truth[i] == False and prediction[i] == True])
#     fn = np.sum([True for i in range(num_samples) \
#                 if ground_truth[i] == True and prediction[i] == False])
#     print(tp, fp, fn)
#     precision = tp / (tp + fp) if (tp + fp) else 0.0
#     recall = tp / (tp + fn)
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
#     accuracy = np.sum(prediction == ground_truth) / num_samples

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = 0
    num_samples = len(prediction)
    accuracy = np.sum(prediction == ground_truth) / num_samples 
    return accuracy
