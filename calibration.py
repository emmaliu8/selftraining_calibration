from math import gamma
from re import I, S
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
import bisect
from typing import List, TypeVar

from uq360.metrics.classification_metrics import expected_calibration_error
from model_training import get_model_predictions
from netcal_package.binning.BBQ import BBQ
from netcal_package.binning.ENIR import ENIR 

# for PlattBinnerCalibrator
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0. 
T = TypeVar('T')

def cross_entropy_loss(probs, true, label_smoothing=False):
    if label_smoothing:
        loss = LabelSmoothingCrossEntropy(epsilon = 0.5)
        return loss(probs, true)
    else:
        loss = LabelSmoothingCrossEntropy(epsilon = 0)
        return loss(probs, true)

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        preds = torch.tensor(preds)
        target = torch.tensor(target)
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon).numpy()

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
        self.label_smoothing = False
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = cross_entropy_loss(scaled_probs, true, self.label_smoothing)
        return loss
    
    # Find the temperature
    def fit(self, logits, true, label_smoothing = False):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        self.label_smoothing = label_smoothing
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
        self.label_smoothing = False
    
    def _ll_w(self, w, p0, p1, p2, label):
        ## find optimal weight coefficients with Cros-Entropy loss function
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = cross_entropy_loss(p, label, self.label_smoothing)
        # ce = -np.sum(label*np.log(p))/N
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

    def fit(self, logits, true, label_smoothing = False):
        self.label_smoothing = label_smoothing
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
            # idx = [element - 1 if element >= len(self.conf) else element for element in idx]
            # probs[i] = [self.conf[j] for j in idx]
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
            # idx = [element - 1 if element >= len(self.conf) else element for element in idx]
            # probs[i] = [self.conf[j] for j in idx]
            probs[i] = self.conf[idx]

        return probs

# class BBQ():
#     def __init__(self, num_binning_models = 10, n_prime = 2.0):
#         self.num_binning_models = num_binning_models
#         self.n_prime = n_prime
#         self.binning_models = []
#         self.binning_model_scores = []

#     def _binning_model_score(self, probs, true, binning_model):
#         num_bins = binning_model.get_num_bins()
#         bin_upper_bounds = binning_model.get_bin_upper_bounds()

#         prob_d_given_m = 1
#         for i in range(int(num_bins)):
#             lower_bound = 0 if i == 0 else bin_upper_bounds[i-1]
#             upper_bound = bin_upper_bounds[i]
#             midpoint = (lower_bound + upper_bound) / 2.0

#             alpha = self.n_prime * midpoint / num_bins
#             beta = self.n_prime * (1 - midpoint) / num_bins

#             labels_in_bin = true[np.argwhere(np.logical_and(probs > lower_bound, probs <= upper_bound))]
#             num_instances_in_bin = len(labels_in_bin)
#             num_positive_class_in_bin = np.count_nonzero(labels_in_bin == 1)
#             num_negative_class_in_bin = num_instances_in_bin - num_positive_class_in_bin

#             first_bin_product = gamma(self.n_prime / num_bins) / gamma(num_instances_in_bin + self.n_prime / num_bins)
#             second_bin_product = gamma(num_positive_class_in_bin + alpha) / gamma(alpha)
#             third_bin_product = gamma(num_negative_class_in_bin + beta) / gamma(beta)
#             total_bin_product = first_bin_product * second_bin_product * third_bin_product
#             prob_d_given_m *= total_bin_product
        
#         return prob_d_given_m # don't need to multiply by p_m bc uniform prior so it cancels out 

#     def fit(self, probs, true):
#         # initialize self.binning_models
#         num_training_samples = len(true)
#         print('in fit')
#         print(num_training_samples)
#         num_bins = int(num_training_samples / 150.0) # to ensure no overflow errors from using gamma function
#         for _ in range(self.num_binning_models):
#             print(num_bins)
#             self.binning_models.append(EqualFreqBinning(int(num_bins)))
#             num_bins += (num_training_samples - int(num_training_samples / 150.0)) / self.num_binning_models

#         for model in self.binning_models:
#             model.fit(probs, true)
#             self.binning_model_scores.append(self._binning_model_score(probs, true, model))
#             # model.predict(probs)
#             plot_calibration_curve(true, model.predict(probs[:1]), 'temperature_scaling_test8/bbq' + model.get_num_bins() + '_calibration.jpg')

#     def predict(self, probs):
#         calibrated_probs = np.zeros(probs.shape)
#         for i in range(self.num_binning_models):
#             calibrated_probs += self.binning_model_scores[i] / np.sum(self.binning_model_scores) * self.binning_models[i].predict(probs)
#         return calibrated_probs

class PlattScaling():
    def __init__(self, a = 1, b = 0, maxiter = 200, solver = "BFGS"):
        self.a = a
        self.b = b
        self.maxiter = maxiter
        self.solver = solver
        self.label_smoothing = False

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x[0], x[1])    
        loss = cross_entropy_loss(scaled_probs, true, self.label_smoothing)
        return loss

    def fit(self, logits, true, label_smoothing = False): # optimize w/ NLL loss according to Guo calibration paper
        self.label_smoothing = label_smoothing
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

class PlattBinnerCalibrator():
    '''
    Borrowed from https://github.com/p-lambda/verified_calibration/blob/master/calibration/calibrators.py#L20
    '''
    def __init__(self, num_bins = 10):
        self._num_bins = num_bins
    
    def get_platt_scaler(self, model_probs, labels):
        clf = LogisticRegression(C=1e10, solver='lbfgs')
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

    def get_equal_bins(self, probs: List[float], num_bins: int=10) -> Bins:
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


# def get_probs_and_labels(dataloader, device, model):
#     logits_list = []
#     labels_list = []

#     for (_, batch) in enumerate(dataloader):
#         inputs, labels = batch['Text'].to(device), batch['Class'].to(device)

#         with torch.no_grad():
#             logits_list.append(model(inputs))
#             labels_list.append(labels)
    
#     logits_list = torch.cat(logits_list).cpu().numpy()
#     labels_list = torch.cat(labels_list).cpu().numpy()

#     return logits_list, labels_list

def calibrate_temperature_scaling(dataloader, device, model, pre_softmax_probs_predict, label_smoothing=False):
    pre_softmax_probs, _, _, _, true_labels = get_model_predictions(dataloader, device, model)

    temperature_scaling = TemperatureScaling()
    temperature_scaling.fit(pre_softmax_probs, true_labels, label_smoothing)
    return temperature_scaling.predict(pre_softmax_probs_predict)

def calibrate_histogram_binning(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        histogram_binning = HistogramBinning()
        y_cal = np.array(true_labels == i, dtype="int")

        if label_smoothing:
            histogram_binning.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        else:
            histogram_binning.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(histogram_binning.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_isotonic_regression(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        isotonic_regression = IsotonicRegression(out_of_bounds='clip')
        y_cal = np.array(true_labels == i, dtype="int")

        if label_smoothing:
            isotonic_regression.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        else:
            isotonic_regression.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(isotonic_regression.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_beta_calibration(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        beta_calibration = BetaCalibration()
        y_cal = np.array(true_labels == i, dtype="int")

        # if label_smoothing:
        #     beta_calibration.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        # else:
        beta_calibration.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(beta_calibration.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_platt_scaling(dataloader, device, model, pre_softmax_probs_predict, label_smoothing=False):
    pre_softmax_probs, _, _, _, true_labels = get_model_predictions(dataloader, device, model)

    platt_scaling = PlattScaling()
    platt_scaling.fit(pre_softmax_probs, true_labels, label_smoothing)
    return platt_scaling.predict(pre_softmax_probs_predict)

def calibrate_equal_freq_binning(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        equal_freq_binning = EqualFreqBinning()
        y_cal = np.array(true_labels == i, dtype="int")

        if label_smoothing:
            equal_freq_binning.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        else:
            equal_freq_binning.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(equal_freq_binning.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_bbq(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        bbq = BBQ()
        y_cal = np.array(true_labels == i, dtype="int")

        if label_smoothing:
            bbq.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        else:
            bbq.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(bbq.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_ensemble_temperature_scaling(dataloader, device, model, pre_softmax_probs_predict, label_smoothing=False):
    pre_softmax_probs, _, _, _, true_labels = get_model_predictions(dataloader, device, model)

    ensemble_temperature_scaling = EnsembleTemperatureScaling()
    ensemble_temperature_scaling.fit(pre_softmax_probs, true_labels, label_smoothing)
    return ensemble_temperature_scaling.predict(pre_softmax_probs_predict)

def calibrate_enir(dataloader, device, model, post_softmax_probs_predict, label_smoothing=False):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        enir = ENIR()
        y_cal = np.array(true_labels == i, dtype="int")

        # if label_smoothing:
        #     enir.fit(post_softmax_probs[:, i], platt_label_smoothing(y_cal))
        # else:
        enir.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(enir.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_platt_binner(dataloader, device, model, pre_softmax_probs_predict, label_smoothing=False):
    pre_softmax_probs, _, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(pre_softmax_probs_predict.shape)

    for i in range(pre_softmax_probs.shape[1]):
        platt_binner = PlattBinnerCalibrator()
        y_cal = np.array(true_labels == i, dtype="int")

        # if label_smoothing:
        #     platt_binner.fit(pre_softmax_probs[:, i], platt_label_smoothing(y_cal))
        # else:
        platt_binner.fit(pre_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(platt_binner.predict(np.expand_dims(pre_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class