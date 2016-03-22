"""
Implements Single Instance Learning SVM
"""
import numpy as np
import inspect
from util import slices
from sklearn.svm import LinearSVC

class LinearSIL(LinearSVC):
    """
    Linear SVC based Single-Instance Learning applied to MI data. The original 
    SIL implementation optimizes the objective function by quadratic programming, 
    making the run-time quadratic in the number of samples. In contrast, this 
    approach uses LinearSVC, a fast linear-time SVM approximation, making it 
    scalable to large amounts of data. This method only works with a linear 
    kernel, however.
    """

    def __init__(self, *args, **kwargs):
        """
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param class_weight : dict, optional
                           Set the parameter C of class i to class_weight[i]*C for
                           SVC. If not given, all classes are supposed to have
                           weight one.
        """
        super(LinearSIL, self).__init__(*args, **kwargs)
        self._bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])
        return super(LinearSIL, self).fit(svm_X, svm_y)

#    def _compute_separator(self, K):
#        super(LinearSIL, self)._compute_separator(K)
#        self._bag_predictions = _inst_to_bag_preds(self._predictions, self._bags)

    def predict(self, bags, instancePrediction = None):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param instancePrediction : flag to indicate if instance predictions 
                                    should be given as output.
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if instancePrediction is None:
            instancePrediction = False
            
        bags = [np.asmatrix(bag) for bag in bags]
        inst_preds = super(LinearSIL, self).predict(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)

    def get_params(self, deep=True):
        """
        return params
        """
        args, _, _, _ = inspect.getargspec(super(LinearSIL, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}


def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(map(len, bags))])
