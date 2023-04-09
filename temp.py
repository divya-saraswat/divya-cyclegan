import os
import numpy as np

# n_classes = 5
# cm = np.array([[10606.4167, 1464.2083, 61.9167, 11.75, 301.4167],
#                [610.5417, 2315.7917, 686.4167, 59.4583, 768.9167],
#                [152.7083, 2469.5833, 8752.2083, 1394.5, 1248.5833],
#                [3.3333, 29.4167, 385.0833, 1920.375, 24.5],
#                [211.9167, 1287.5833, 614.7083, 71.875, 2945.3333]])
n_classes = 3
cm = np.array([[456, 0, 26],
               [0, 236, 8],
               [700, 80, 0]]).T
print(cm)
# np.set_printoptions(suppress=True, precision=4)


for c in range(n_classes):
    tp = cm[c, c]
    fp = sum(cm[:, c]) - cm[c, c]
    fn = sum(cm[c, :]) - cm[c, c]
    tn = sum(np.delete(sum(cm) - cm[c, :], c))

    print("Class {}, False Positive: {}".format(c, round(fp / (tp + fp + fn + tn), 4)))
    print("Class {}, False Negative: {}".format(c, round(fn / (tp + fp + fn + tn), 4)))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print(f"for class {c}: acc {accuracy}, recall {recall},\
         precision {precision}, f1 {f1_score}")
    print("for class {}: recall {}, specificity {}\
          precision {}, accuracy {}, f1 {}".format(c, round(recall, 4), round(specificity, 4), round(precision, 4),
                                                   round(accuracy, 4),
                                                   round(f1_score, 4)))
