from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
import bisect
from typing import List, TypeVar

from uq360.metrics.classification_metrics import expected_calibration_error
import model_training
from netcal_package.binning.BBQ import BBQ
from netcal_package.binning.ENIR import ENIR 
# from keras.utils import to_categorical
# from dirichlet_python.dirichletcal.calib.vectorscaling import VectorScaling1
# from dirichlet_python.dirichletcal.calib.matrixscaling import MatrixScaling1
# from dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator

# for PlattBinnerCalibrator
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0. 
T = TypeVar('T')

def to_categorical(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1 # assumes labels are 0-indexed
    labels_list = list(labels)
    result = np.zeros((len(labels_list), num_classes))
    for i in range(len(labels_list)):
        result[i] = np.eye(num_classes, dtype='uint8')[labels_list[i]]
    return result

def plot_calibration_curve(y_true, y_prob, filename, num_bins=10):
    if len(y_prob.shape) == 1:
        y_prob = y_prob.reshape(-1, 1)

    ece, confidences_in_bins, accuracies_in_bins, frac_samples_in_bins, bin_centers = expected_calibration_error(y_true, y_prob, num_bins=num_bins, return_counts=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(bin_centers, frac_samples_in_bins, 'o-')
    plt.title("Confidence Histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Fraction of Samples")
    plt.grid()
    plt.ylim([0.0, 1.0])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, accuracies_in_bins, 'o-',
                 label="ECE = {:.2f}".format(ece))
    plt.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50), 'b.', label="Perfect Calibration")
    plt.title("Reliability Plot")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.savefig(filename)

    return ece, confidences_in_bins, accuracies_in_bins, frac_samples_in_bins, bin_centers

def softmax(x):
    """
    Borrowed from https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py

    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)

class LogitRegression(LinearRegression):
    '''
    from https://stackoverflow.com/questions/44234682/how-to-use-sklearn-when-target-variable-is-a-proportion
    '''

    def fit(self, x, p):
        p = np.asarray(p)
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

def temp_log_loss(y_true, y_pred):
    # y_pred = check_array(y_pred, ensure_2d=False)
    # check_consistent_length(y_pred, y_true, sample_weight)

    # lb = LabelBinarizer()

    # if labels is not None:
    #     lb.fit(labels)
    # else:
    #     lb.fit(y_true)

    # if len(lb.classes_) == 1:
    #     if labels is None:
    #         raise ValueError(
    #             "y_true contains only one label ({0}). Please "
    #             "provide the true labels explicitly through the "
    #             "labels argument.".format(lb.classes_[0])
    #         )
    #     else:
    #         raise ValueError(
    #             "The labels array needs to contain at least two "
    #             "labels for log_loss, "
    #             "got {0}.".format(lb.classes_)
    #         )

    # transformed_labels = lb.transform(y_true)
    transformed_labels = y_true.reshape(-1, 1)
    if transformed_labels.shape[1] == 1 and y_pred.shape[1] == 2:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )
    elif transformed_labels.shape[1] == 1 and y_pred.shape[1] > 2:
        transformed_labels = to_categorical(y_true)

    # Clipping
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    # transformed_labels = check_array(transformed_labels)
    # if len(lb.classes_) != y_pred.shape[1]:
    #     if labels is None:
    #         raise ValueError(
    #             "y_true and y_pred contain different number of "
    #             "classes {0}, {1}. Please provide the true "
    #             "labels explicitly through the labels argument. "
    #             "Classes found in "
    #             "y_true: {2}".format(
    #                 transformed_labels.shape[1], y_pred.shape[1], lb.classes_
    #             )
    #         )
    #     else:
    #         raise ValueError(
    #             "The number of classes in labels is different "
    #             "from that in y_pred. Classes found in "
    #             "labels: {0}".format(lb.classes_)
    #         )

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight=None, normalize=True)

class TemperatureScaling():
    '''
    Borrowed from https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    '''
    
    def __init__(self, temp = 1, maxiter = 200, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)   
        loss = temp_log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)

class EnsembleTemperatureScaling():
    # code from https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/util_calibration.py
    def __init__(self):
        self.t = None
        self.w = None
    
    def _ll_w(self, w, p0, p1, p2, label):
        ## find optimal weight coefficients with Cros-Entropy loss function
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = temp_log_loss(label, p)
        return ce

    def _fit_ensemble_scaling(self, logits, true, temp, n_class=2):
        p1 = np.exp(logits)/np.sum(np.exp(logits), 1)[:, None]
        logits = logits/temp
        p0 = np.exp(logits)/np.sum(np.exp(logits), 1)[:, None]
        p2 = np.ones_like(p0)/n_class
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = {"type": "eq", "fun": my_constraint_fun,}
        w = minimize(self._ll_w, x0 = (1.0, 0.0, 0.0) , args = (p0, p1, p2, true), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12)
        w = w.x
        return w

    def fit(self, logits, true):
        temp_scaling = TemperatureScaling()
        temp_scaling.fit(logits, true)
        self.t = temp_scaling.temp
        self.w = self._fit_ensemble_scaling(logits, true, self.t)

    def predict(self, logits, n_class=2):
        p1 = np.exp(logits) / np.sum(np.exp(logits), 1)[:, None]
        logits = logits / self.t 
        p0 = np.exp(logits) / np.sum(np.exp(logits), 1)[:, None]
        p2 = np.ones_like(p0) / n_class
        p = self.w[0] * p0 + self.w[1] * p1 + self.w[2] * p2 
        return p

class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=50):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    def get_num_bins(self):
        return len(self.upper_bounds)
    
    def get_bin_upper_bounds(self):
        return self.upper_bounds[:]

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """
        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf

    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """
        # Go through all the probs and check what confidence is suitable for it.
        probs = np.copy(probs)
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob[0])
            probs[i] = self.conf[idx]

        return probs

class EqualFreqBinning():
    def __init__(self, num_bins = 50):
        self.num_bins = num_bins
        self.conf = []
        self.upper_bounds = []
    
    def get_num_bins(self):
        return self.num_bins
    
    def get_bin_upper_bounds(self):
        return self.upper_bounds[:]

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    
    
    def fit(self, probs, true):
        sorted_probs_indices = np.argsort(probs)
        sorted_probs = probs[sorted_probs_indices]
        sorted_true = true[sorted_probs_indices]
        split_probs = np.array_split(sorted_probs, self.num_bins)
        split_true = np.array_split(sorted_true, self.num_bins)
        
        # update self.upper_bounds
        for i in range(len(split_probs)-1):
            self.upper_bounds.append((split_probs[i][-1] + split_probs[i+1][0]) / 2.0)
            # IndexError: index 0 is out of bounds for axis 0 with size 0
        self.upper_bounds.append(1)

        # update self.conf
        conf = []

        for i in range(1, len(self.upper_bounds) + 1):
            lower = 0 if i == 1 else self.upper_bounds[i-1]
            upper = 1 if i == len(self.upper_bounds) else self.upper_bounds[i]
            temp_conf = self._get_conf(lower, upper, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    def predict(self, probs):
        probs = np.copy(probs)
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob[0])
            probs[i] = self.conf[idx]

        return probs

class BetaCalibration():
    '''
    Code borrowed from https://github.com/betacal/python/blob/master/betacal/beta_calibration.py
    '''
    def __init__(self):
        self.map_ = None 
        self.lr_ = None
    
    def _beta_calibration(self, df, y):
        df = df.reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1

        if np.unique(y)[0] == 0 and np.unique(y)[1] == 1:
            lr = LogisticRegression(C=99999999999)
            lr.fit(x, y)
            coefs = lr.coef_[0]
        else:
            lr = LogitRegression()
            lr.fit(x, y)
            coefs = lr.coef_

        if coefs[0] < 0:
            x = x[:, 1].reshape(-1, 1)
            if np.unique(y)[0] == 0 and np.unique(y)[1] == 1:
                lr = LogisticRegression(C=99999999999)
            else:
                lr = LogitRegression()
            lr.fit(x, y)
            coefs = lr.coef_[0]
            a = 0
            b = coefs[0]
        elif coefs[1] < 0:
            x = x[:, 0].reshape(-1, 1)
            if np.unique(y)[0] == 0 and np.unique(y)[1] == 1:
                lr = LogisticRegression(C=99999999999)
            else:
                lr = LogitRegression()
            lr.fit(x, y)
            coefs = lr.coef_[0]
            a = coefs[0]
            b = 0
        else:
            a = coefs[0]
            b = coefs[1]

        if np.unique(y)[0] == 0 and np.unique(y)[1] == 1:
            inter = lr.intercept_[0]
        else:
            inter = lr.intercept_
        
        m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                            bounds=[0, 1], method='Bounded').x
        map = [a, b, m]
        return map, lr

    def fit(self, probs, true):
        self.map_, self.lr_ = self._beta_calibration(probs, true)

    def predict(self, probs):
        df = probs.reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        if hasattr(self.lr_, 'predict_proba'):
            return self.lr_.predict_proba(x)[:, 1]
        else:
            return self.lr_.predict(x)

class PlattScaling():
    def __init__(self, a = 1, b = 0, maxiter = 200, solver = "BFGS"):
        self.a = a
        self.b = b
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x[0], x[1])  
        loss = temp_log_loss(true, scaled_probs)
        return loss

    def fit(self, logits, true): # optimize w/ NLL loss according to Guo calibration paper
        true = true.flatten()
        opt = minimize(self._loss_fun, x0 = [0.55, 0], args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.a = opt.x[0]
        self.b = opt.x[1]
        return opt

    def predict(self, logits, a = None, b = None):
        if not a or not b:
            return softmax(self.a * logits + self.b)
        else:
            return softmax(a * logits + b)

class VectorScaling():
    def __init__(self, n_classes, a_diags = None, b = None, maxiter = 200, solver = "BFGS"):
        self.n_classes = n_classes
        if a_diags is None:
            self.a_diags = [1 for _ in range(n_classes)]
        else: 
            if len(a_diags) != n_classes:
                raise ValueError("Length of a_diags should be equal to n_classes")
            else:
                self.a_diags = a_diags
        # use self.a_diags to make a -> shape: (n_classes, n_classes) - diagonal matrix
        if b is None:
            self.b = np.zeros((n_classes, 1)) # shape: (n_classes, 1)
        else:
            if b.shape[0] != n_classes or b.shape[1] != 1:
                raise ValueError("shape of b should be (n_classes, 1)")
            else:
                self.b = b
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        a_diags = x[:self.n_classes]
        b = np.reshape(x[self.n_classes:], (self.n_classes, 1))
        scaled_probs = self.predict(probs, a_diags, b)  
        loss = temp_log_loss(true, scaled_probs)
        return loss

    def fit(self, logits, true): # optimize w/ NLL loss according to Guo calibration paper
        true = true.flatten()
        initial_a_diags = [1 for _ in range(self.n_classes)]
        initial_b = np.zeros((self.n_classes, 1))
        initial_x0 = initial_a_diags + list(initial_b.flatten())
        opt = minimize(self._loss_fun, x0 = initial_x0, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.a_diags = opt.x[0:self.n_classes]
        self.b = np.reshape(opt.x[self.n_classes:], (self.n_classes, 1))
        return opt

    def predict(self, logits, a_diags = None, b = None):
        if a_diags is None or b is None:
            a = np.diag(self.a_diags)
            return softmax(np.matmul(a, logits.T) + self.b).T
        else:
            a = np.diag(a_diags)
            return softmax(np.matmul(a, logits.T) + b).T

class MatrixScaling():
    def __init__(self, n_classes, a = None, b = None, maxiter = 200, solver = "BFGS"):
        self.n_classes = n_classes
        self.a = np.diag([a for _ in range(n_classes)]) # shape: (n_classes, n_classes)
        self.b = np.full((n_classes, 1), b) # shape: (n_classes, 1)
        if a is None:
            pass 
        else:
            if a.shape[0] != n_classes or a.shape[1] != n_classes:
                raise ValueError("shape of a should be (n_classes, n_classes)")
            else:
                self.a = a
        if b is None:
            pass 
        else:
            if b.shape[0] != n_classes or b.shape[1] != n_classes:
                raise ValueError("shape of b should be (n_classes, 1)")
            else:
                self.b = b
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        # use x to get a and b
        # first self.n_classes * self.n_classes params for a
        # last self.n_classes params for b
        a = np.reshape(np.array(x[:self.n_classes*self.n_classes]), (self.n_classes, self.n_classes))
        b = np.reshape(np.array(x[self.n_classes*self.n_classes:]), (self.n_classes, 1))
        scaled_probs = self.predict(probs, a, b)  
        loss = temp_log_loss(true, scaled_probs)
        return loss

    def fit(self, logits, true): # optimize w/ NLL loss according to Guo calibration paper
        true = true.flatten()
        initial_a = np.eye(self.n_classes)
        initial_b = np.zeros((self.n_classes, 1))
        x0 = list(initial_a.flatten()) + list(initial_b.flatten())
        opt = minimize(self._loss_fun, x0 = x0, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.a = np.reshape(np.array(opt.x[:self.n_classes*self.n_classes]), (self.n_classes, self.n_classes))
        self.b = np.reshape(np.array(opt.x[self.n_classes*self.n_classes:]), (self.n_classes, 1))
        return opt

    def predict(self, logits, a = None, b = None):
        if a is None or b is None:
            return softmax(np.matmul(self.a, logits.T) + self.b).T
        else:
            return softmax(np.matmul(a, logits.T) + b).T

class PlattBinnerCalibrator():
    '''
    Borrowed from https://github.com/p-lambda/verified_calibration/blob/master/calibration/calibrators.py#L20
    '''
    def __init__(self, num_bins = 10):
        self._num_bins = num_bins
    
    def get_platt_scaler(self, model_probs, labels):
        if np.unique(labels)[0] == 0 and np.unique(labels)[1] == 1:
            clf = LogisticRegression(C=1e10, solver='lbfgs')
        else:
            clf = LogitRegression()
        eps = 1e-12
        model_probs = model_probs.astype(dtype=np.float64)
        model_probs = np.expand_dims(model_probs, axis=-1)
        model_probs = np.clip(model_probs, eps, 1 - eps)
        model_probs = np.log(model_probs / (1 - model_probs))
        clf.fit(model_probs, labels)
        def calibrator(probs):
            x = np.array(probs, dtype=np.float64)
            x = np.clip(x, eps, 1 - eps)
            x = np.log(x / (1 - x))
            x = x * clf.coef_[0] + clf.intercept_
            output = 1 / (1 + np.exp(-x))
            return output 
        return calibrator
    
    def split(self, sequence: List[T], parts: int) -> List[List[T]]:
        assert parts <= len(sequence)
        part_size = int(np.ceil(len(sequence) * 1.0 / parts))
        assert part_size * parts >= len(sequence)
        assert (part_size - 1) * parts < len(sequence)
        return [sequence[i:i + part_size] for i in range(0, len(sequence), part_size)]

    def get_equal_bins(self, probs, num_bins: int=10) -> Bins:
        """Get bins that contain approximately an equal number of data points."""
        sorted_probs = sorted(probs)
        binned_data = self.split(sorted_probs, num_bins)
        bins: Bins = []
        for i in range(len(binned_data) - 1):
            last_prob = binned_data[i][-1]
            next_first_prob = binned_data[i + 1][0]
            bins.append((last_prob + next_first_prob) / 2.0)
        bins.append(1.0)
        bins = sorted(list(set(bins)))
        return bins
    
    def get_bin(self, pred_prob: float, bins: List[float]) -> int:
        """Get the index of the bin that pred_prob belongs in."""
        assert 0.0 <= pred_prob <= 1.0
        assert bins[-1] == 1.0
        return bisect.bisect_left(bins, pred_prob)
    
    def get_histogram_calibrator(self, model_probs, values, bins):
        binned_values = [[] for _ in range(len(bins))]
        for prob, value in zip(model_probs, values):
            bin_idx = self.get_bin(prob, bins)
            binned_values[bin_idx].append(float(value))
        def safe_mean(values, bin_idx):
            if len(values) == 0:
                if bin_idx == 0:
                    return float(bins[0]) / 2.0
                return float(bins[bin_idx] + bins[bin_idx - 1]) / 2.0
            return np.mean(values)
        bin_means = [safe_mean(values, bidx) for values, bidx in zip(binned_values, range(len(bins)))]
        bin_means = np.array(bin_means)
        def calibrator(probs):
            indices = np.searchsorted(bins, probs)
            return bin_means[indices]
        return calibrator
    
    def get_discrete_calibrator(self, model_probs, bins):
        return self.get_histogram_calibrator(model_probs, model_probs, bins)

    def fit(self, zs, ys):
        self._platt = self.get_platt_scaler(zs, ys)
        platt_probs = self._platt(zs)
        bins = self.get_equal_bins(platt_probs, num_bins=self._num_bins)
        self._discrete_calibrator = self.get_discrete_calibrator(platt_probs, bins)

    def predict(self, zs):
        platt_probs = self._platt(zs)
        return self._discrete_calibrator(platt_probs)
    
def platt_label_smoothing(labels):
    num_positive = np.count_nonzero(labels == 1)
    num_negative = len(labels) - num_positive
    new_positive_label = 1.0 * (num_positive + 1) / (num_positive + 2)
    new_negative_label = 1.0 / (num_negative + 2)
    new_labels = np.where(labels == 1, new_positive_label, new_negative_label)
    return new_labels

def alpha_label_smoothing(labels, alpha):
    new_positive_label = (1.0 - alpha) + alpha / 2.0
    new_negative_label = alpha / 2.0
    new_labels = np.where(labels == 1, new_positive_label, new_negative_label)
    return new_labels

def apply_label_smoothing(labels, label_smoothing, alpha = 0.1):
    if label_smoothing == 'none':
        return labels
    elif label_smoothing == 'platt':
        return platt_label_smoothing(labels)
    elif label_smoothing == 'alpha':
        return alpha_label_smoothing(labels, alpha)
    else:
        raise ValueError("Invalid method of label smoothing")

def calibrate_temperature_scaling(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models, use_post_softmax=True)

    temperature_scaling = TemperatureScaling()
    temperature_scaling.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
    return temperature_scaling.predict(pre_softmax_probs_predict)

def calibrate_histogram_binning(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)
        
    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        histogram_binning = HistogramBinning()
        y_cal = np.array(true_labels == i, dtype="int")
        histogram_binning.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(histogram_binning.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_isotonic_regression(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        isotonic_regression = IsotonicRegression(out_of_bounds='clip')
        y_cal = np.array(true_labels == i, dtype="int")
        isotonic_regression.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(isotonic_regression.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_beta_calibration(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        beta_calibration = BetaCalibration()
        y_cal = np.array(true_labels == i, dtype="int")
        beta_calibration.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(beta_calibration.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_platt_scaling(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    platt_scaling = PlattScaling()
    platt_scaling.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
    return platt_scaling.predict(pre_softmax_probs_predict)

def calibrate_equal_freq_binning(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        equal_freq_binning = EqualFreqBinning()
        y_cal = np.array(true_labels == i, dtype="int")
        equal_freq_binning.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(equal_freq_binning.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_bbq(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        bbq = BBQ()
        y_cal = np.array(true_labels == i, dtype="int")
        bbq.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(bbq.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_ensemble_temperature_scaling(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    ensemble_temperature_scaling = EnsembleTemperatureScaling()
    ensemble_temperature_scaling.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
    return ensemble_temperature_scaling.predict(pre_softmax_probs_predict)

def calibrate_enir(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        enir = ENIR()
        y_cal = np.array(true_labels == i, dtype="int")

        if label_smoothing is not 'none':
            raise ValueError('Label smoothing not supported for ENIR')
        else:
            enir.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(enir.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_platt_binner(dataloader, device, models, post_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None):
    _, post_softmax_probs, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        platt_binner = PlattBinnerCalibrator()
        y_cal = np.array(true_labels == i, dtype="int")
        platt_binner.fit(post_softmax_probs[:, i], apply_label_smoothing(y_cal, label_smoothing, label_smoothing_alpha))
        calibrated_probs_by_class[:, i] = np.squeeze(platt_binner.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_vector_scaling(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None, num_classes=2):
    pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models, use_post_softmax=True)

    vector_scaling = VectorScaling(num_classes)
    vector_scaling.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
    return vector_scaling.predict(pre_softmax_probs_predict)

def calibrate_matrix_scaling(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None, num_classes=2):
    pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models, use_post_softmax=True)

    matrix_scaling = MatrixScaling(num_classes)
    matrix_scaling.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
    return matrix_scaling.predict(pre_softmax_probs_predict)

# def calibrate_full_dirichlet(dataloader, device, models, pre_softmax_probs_predict, label_smoothing='none', label_smoothing_alpha=None, num_classes=2):
#     pre_softmax_probs, _, _, _, true_labels, _ = model_training.get_model_predictions(dataloader, device, models, use_post_softmax=True)

#     full_dirichlet = FullDirichletCalibrator()
#     full_dirichlet.fit(pre_softmax_probs, apply_label_smoothing(true_labels, label_smoothing, label_smoothing_alpha))
#     return full_dirichlet.predict(pre_softmax_probs_predict)

