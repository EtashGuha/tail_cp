import sys
import torch
import numpy as np
import cqr_helpers.torch_models as torch_models
from functools import partial
from cqr_helpers.nonconformist.cp import IcpRegressor
from cqr_helpers.nonconformist.base import RegressorAdapter
import datetime

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    y_test = np.asarray(y_test)
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length

def run_icp(nc, X_train, y_train, X_cal, y_cal, X_test, idx_train, idx_cal, significance, condition=None):
    """ Run split conformal method

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    icp = IcpRegressor(nc,condition=condition)

    # Fit the ICP using the proper training set
    icp.fit(X_train[idx_train,:], y_train[idx_train])

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_train[idx_cal,:], y_train[idx_cal])

    # Produce predictions for the test set, with confidence 90%
    predictions = icp.predict(X_test, significance=significance)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    return y_lower, y_upper

def compute_coverage(y_test,y_lower,y_upper,significance,name=""):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_coverage = np.mean(in_the_range) * 100
    std_coverage = np.std(in_the_range)
    print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance*100, coverage))
    sys.stdout.flush()

    avg_length = abs(np.mean(y_lower - y_upper))
    std_length = abs(np.std(y_upper - y_lower))
    print("%s: Average length: %f" % (name, avg_length))
    sys.stdout.flush()
    return avg_coverage, std_coverage, avg_length, std_length

###############################################################################
# Deep neural network for conditional quantile regression
# Minimizing pinball loss
###############################################################################

class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)
