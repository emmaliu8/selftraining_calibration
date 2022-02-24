from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import numpy as np
import torch
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration

from uq360.metrics.classification_metrics import expected_calibration_error
from model_training import get_model_predictions

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
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
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

def calibrate_temperature_scaling(dataloader, device, model, pre_softmax_probs_predict):
    pre_softmax_probs, _, _, _, true_labels = get_model_predictions(dataloader, device, model)

    temperature_scaling = TemperatureScaling()
    temperature_scaling.fit(pre_softmax_probs, true_labels)

    return temperature_scaling.predict(pre_softmax_probs_predict)

def calibrate_histogram_binning(dataloader, device, model, post_softmax_probs_predict):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        histogram_binning = HistogramBinning()
        y_cal = np.array(true_labels == i, dtype="int")
        histogram_binning.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(histogram_binning.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_isotonic_regression(dataloader, device, model, post_softmax_probs_predict):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        isotonic_regression = IsotonicRegression()
        y_cal = np.array(true_labels == i, dtype="int")
        isotonic_regression.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(isotonic_regression.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class

def calibrate_beta_calibration(dataloader, device, model, post_softmax_probs_predict):
    _, post_softmax_probs, _, _, true_labels = get_model_predictions(dataloader, device, model)

    calibrated_probs_by_class = np.zeros(post_softmax_probs_predict.shape)

    for i in range(post_softmax_probs.shape[1]):
        beta_calibration = BetaCalibration()
        y_cal = np.array(true_labels == i, dtype="int")
        beta_calibration.fit(post_softmax_probs[:, i], y_cal)
        calibrated_probs_by_class[:, i] = np.squeeze(beta_calibration.predict(np.expand_dims(post_softmax_probs_predict[:, i], 1)))

    return calibrated_probs_by_class


