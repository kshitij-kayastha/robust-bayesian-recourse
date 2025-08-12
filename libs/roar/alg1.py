import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class LAROAR(object):
    """Class for generate counterfactual samples for framework: ROAR"""

    DECISION_THRESHOLD = 0.5

    def __init__(
        self,
        data,
        coef,
        intercept,
        cat_indices=list(),
        lambd=0.1,
        delta_min=None,
        delta_max=0.1,
        lr=0.5,
        dist_type=1,
        max_iter=20,
        encoding_constraints=False,
    ):
        self.data = data
        self.bias = np.array([intercept / np.linalg.norm(coef, 2)])
        self.weights = coef / np.linalg.norm(coef, 2)
        self.cat_indices = cat_indices
        self.lamb = lambd
        self.lr = lr
        self.dist_type = dist_type
        self.delta_min = delta_min
        self.alpha = delta_max
        self.max_iter = max_iter
        self.encoding_constraints = encoding_constraints
        self.y_target = 1.
    
    def calc_delta(self, w: float, c: float):
        if (w > self.lamb):
            delta = ((np.log((w - self.lamb)/self.lamb) - c) / w)
            if delta < 0: delta = 0.
        elif (w < -self.lamb):
            delta = (np.log((-w - self.lamb)/self.lamb) - c) / w
            if delta > 0: delta = 0.
        else:
            delta = 0.
        return delta   
    
    def can_change_sign(self, w: float):
        return np.sign(w+self.alpha) != np.sign(w-self.alpha)

    def get_max_idx(self, weights: np.ndarray, changed: List):
        weights_copy = deepcopy(weights)
        while True:
            idx = np.argmax(np.abs(weights_copy))
            if not changed[idx]:
                return idx
            else:
                weights_copy[idx] = 0.

    def calc_theta_adv(self, x: np.ndarray):
        weights_adv = self.weights - (np.sign(x) * self.alpha)
        for i in range(len(x)):
            if np.sign(x[i]) == 0:
                weights_adv[i] = weights_adv[i] - (np.sign(weights_adv[i]) * self.alpha)
        bias_adv = self.bias - self.alpha
        
        return weights_adv, bias_adv

    def fit_instance(self, x_0, verbose=False):
        x = deepcopy(x_0)
        weights, bias = self.calc_theta_adv(x)
        changed = [False] * len(weights)
        while True:
           if np.all(changed):
               break
           
           i = self.get_max_idx(weights, changed)
           x_i, w_i = x[i], weights[i]
           
           c = np.matmul(x, weights) + bias
           delta = self.calc_delta(w_i, c[0])
           
           if delta == 0:
               break        
           if (np.sign(x_i+delta) == np.sign(x_i)) or (x_i == 0):
               x[i] = x_i + delta
               changed[i] = True
           else:
               x[i] = 0
               weights[i] = self.weights[i] + (np.sign(x_i) * self.alpha)
               if self.can_change_sign(self.weights[i]):
                   changed[i] = True
        
        
        w = torch.from_numpy(self.weights.copy()).float()
        b = torch.tensor(self.bias).float()
        f_x = torch.sigmoid(torch.dot(torch.from_numpy(x.copy()).float(), w) + b).float()
        self.feasible = f_x.data.item() > 0.5
        
        return x
