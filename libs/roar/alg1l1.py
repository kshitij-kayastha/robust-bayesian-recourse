import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List
import cvxpy as cp

class L1Recourse(object):
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
        # self.bias = np.array([intercept])
        # self.weights = coef
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

        if coef is not None and intercept is not None:
            self._theta0 = np.concat((self.weights, self.bias))
            self._theta0_pt = torch.from_numpy(self._theta0)

    def set_weights(self, weights):
        self.weights = weights
        if self.bias is not None:
            self._theta0 = np.concat((self.weights, self.bias))
            self._theta0_pt = torch.from_numpy(self._theta0)

    def set_bias(self, bias):
        self.bias = bias
        if self.weights is not None:
            self._theta0 = np.concat((self.weights, self.bias))
            self._theta0_pt = torch.from_numpy(self._theta0)

    def generateThetas(self):
        thetas = self._theta0.copy()
        if self.alpha == 0:
            return np.array([thetas])
        
        thetas = np.repeat(thetas.reshape(1, self._theta0.size), (self._theta0.size * 2) - 1, axis=0)
        thetas_i = 0

        for i in range(self._theta0.size):
            if i == self._theta0.size - 1:
                thetas[thetas_i][i] -= self.alpha
                thetas_i += 1
                break

            thetas[thetas_i][i] += self.alpha
            thetas_i += 1
            thetas[thetas_i][i] -= self.alpha
            thetas_i += 1

        return thetas
    
    def getStats(self, x0: np.ndarray, x: np.ndarray, theta: np.ndarray):
        return np.log(1 + np.exp(-(x @ theta))) + \
            (self.lamb * (np.linalg.norm(x0 - x, ord=1)))

    def isFeasible(self, x: torch.Tensor, thetaP: torch.Tensor, has_bias=True):
        attacked_i = torch.argmax(torch.abs(thetaP - self._theta0_pt))
        if not has_bias:
            x = torch.cat((x, torch.tensor([1])))
        maxX_i = torch.argmax(torch.abs(x))
        
        if torch.abs(x[attacked_i]) >= torch.abs(x[maxX_i]):
            # Bias
            if attacked_i == len(x) - 1:
                return True

            if torch.sign(thetaP[attacked_i] - self._theta0_pt[attacked_i]) > 0:
                if torch.sign(x[attacked_i]) <= 0:
                    return True
            else:
                if torch.sign(x[attacked_i]) >= 0:
                    return True
        return False
    
    def getConstraints(self, thetaP: np.ndarray):
        attacked_i = np.argmax(np.abs(thetaP - self._theta0))

        if attacked_i == (thetaP.size - 1):
            A = None
            b = None
        else:
            A = np.zeros(((2 * self._theta0.size) - 3, self._theta0.size - 1))
            b = np.zeros(A.shape[0])

            A[:, attacked_i] = 1
            A_i = 1
            b[0] = -1

            for i in range(A.shape[1]):
                if i == attacked_i:
                    continue
                
                A[A_i, i] = 1
                A_i += 1
                A[A_i, i] = -1
                A_i += 1

            if np.sign(thetaP[attacked_i] - self._theta0[attacked_i]) < 0:
                A[:, attacked_i] *= -1

        return A, b
    
    def projectToF(self, xR: np.ndarray, A: np.ndarray, b: np.ndarray, has_bias=True):
        if has_bias:
            xR = xR[:-1]
            
        x = cp.Variable(xR.size)
        objective = cp.Minimize(0.5 * cp.sum_squares(xR - x))
        constraints = [A @ x <= b]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return x.value
    
    def _runPSDInvScalingMainDs(self, x0_pt: torch.Tensor, thetaP: torch.Tensor, A, b, abstol:float = 1e-8, 
                                  n_epochs:int = 2000, lr0:float = 5, power_t:float = 0.5):

        if not self.isFeasible(x0_pt, thetaP, has_bias=True):
            xR = torch.zeros(len(x0_pt) - 1, dtype=torch.float64, requires_grad=True)
        else:
            xR = x0_pt[:-1].clone().requires_grad_(True)

        loss = torch.tensor(1.)
        loss_diff = 1.

        optimizer = torch.optim.SGD([xR], lr=lr0)
        lr_lambda = lambda last_epoch: (last_epoch + 1) ** (-power_t)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(n_epochs):
            if loss_diff <= abstol:
                break

            loss_prev = loss.clone().detach()
            optimizer.zero_grad()

            f_x = torch.nn.Sigmoid()(torch.matmul(xR, thetaP[:-1]) + thetaP[-1])
            bce_loss = torch.nn.BCELoss()(f_x.unsqueeze(0), torch.ones(1, dtype=torch.float64))
            cost = torch.dist(x0_pt[:-1], xR, 1)
            loss = bce_loss + self.lamb*cost

            loss.backward()
            optimizer.step()
            scheduler.step()

            if not self.isFeasible(xR.detach().clone(), thetaP, has_bias=False):
                xR.data = torch.tensor(self.projectToF(xR.detach().clone().numpy(), A, b, has_bias=False))
                
            loss_diff = torch.dist(loss_prev, loss, 1)
        
        return np.append(xR.detach().numpy(), 1)

    def _runPSDInvScalingBias(self, x0_pt: torch.Tensor, thetaP: torch.Tensor, abstol:float = 1e-8, 
                                  n_epochs:int = 2000, lr0:float = 5, power_t:float = 0.5):
        
        if not self.isFeasible(x0_pt, thetaP, has_bias=True):
            xR = torch.zeros(len(x0_pt) - 1, dtype=torch.float64, requires_grad=True)
        else:
            xR = x0_pt[:-1].clone().requires_grad_(True)

        loss = torch.tensor(1.)
        loss_diff = 1.

        optimizer = torch.optim.SGD([xR], lr=lr0)
        lr_lambda = lambda last_epoch: (last_epoch + 1) ** (-power_t)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(n_epochs):
            if loss_diff <= abstol:
                break

            loss_prev = loss.clone().detach()
            optimizer.zero_grad()

            f_x = torch.nn.Sigmoid()(torch.matmul(xR, thetaP[:-1]) + thetaP[-1])
            bce_loss = torch.nn.BCELoss()(f_x.unsqueeze(0), torch.ones(1, dtype=torch.float64))
            cost = torch.dist(x0_pt[:-1], xR, 1)
            loss = bce_loss + self.lamb*cost

            loss.backward()
            optimizer.step()
            scheduler.step()

            if not self.isFeasible(xR.detach().clone(), thetaP, has_bias=False):
                xR.data = xR.data.clamp(min= -1, max= 1)
                
            loss_diff = torch.dist(loss_prev, loss, 1)

        return np.append(xR.detach().numpy(), 1)
    
    def _runPSDInvScalingAlphaZero(self, x0_pt: torch.Tensor, thetaP: torch.Tensor, abstol:float = 1e-8, 
                                  n_epochs:int = 2000, lr0:float = 5, power_t:float = 0.5):
        
        xR = x0_pt[:-1].clone().requires_grad_(True)

        loss = torch.tensor(1.)
        loss_diff = 1.

        optimizer = torch.optim.SGD([xR], lr=lr0)
        lr_lambda = lambda last_epoch: (last_epoch + 1) ** (-power_t)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(n_epochs):
            if loss_diff <= abstol:
                break

            loss_prev = loss.clone().detach()
            optimizer.zero_grad()

            f_x = torch.nn.Sigmoid()(torch.matmul(xR, thetaP[:-1]) + thetaP[-1])
            bce_loss = torch.nn.BCELoss()(f_x.unsqueeze(0), torch.ones(1, dtype=torch.float64))
            cost = torch.dist(x0_pt[:-1], xR, 1)
            loss = bce_loss + self.lamb*cost

            loss.backward()
            optimizer.step()
            scheduler.step()
                
            loss_diff = torch.dist(loss_prev, loss, 1)

        return np.append(xR.detach().numpy(), 1)

    def runPSDInvScaling(self, x0: np.ndarray, thetaP: np.ndarray, abstol:float = 1e-8, 
                                  n_epochs:int = 2000, lr0:float = 5, power_t:float = 0.5):
        
        x0_pt = torch.from_numpy(x0)
        
        attacked_i = np.argmax(np.abs(thetaP - self._theta0))
        thetaP_pt = torch.from_numpy(thetaP)

        if (thetaP == self._theta0).all():
            xR = self._runPSDInvScalingAlphaZero(x0_pt, thetaP_pt, abstol, n_epochs, lr0, power_t) 
        elif attacked_i == (thetaP.size - 1):
            xR = self._runPSDInvScalingBias(x0_pt, thetaP_pt, abstol, n_epochs, lr0, power_t)
        else:
            A, b = self.getConstraints(thetaP)
            xR = self._runPSDInvScalingMainDs(x0_pt, thetaP_pt, A, b, abstol, n_epochs, lr0, power_t)

        return xR
    
    def runPSDInvScalingAllThetas(self, x0: np.ndarray, abstol:float = 1e-8, n_epochs:int = 2000, 
                                    lr0:float = 5, power_t:float = 0.5, returnDataFrame = False):
        
        thetaPs = self.generateThetas()
        xRs = np.empty(thetaPs.shape)
        Js = np.empty(thetaPs.shape[0])

        for i, thetaP in enumerate(thetaPs):
            xRs[i] = self.runPSDInvScaling(x0, thetaP, abstol, n_epochs, lr0, power_t)
            Js[i] = self.getStats(x0, xRs[i], thetaP)

        if returnDataFrame:
            everything = [(xRs[i], thetaPs[i], x0, self._theta0, Js[i], self.alpha, self.lamb) for i in range(len(Js))]
            df = pd.DataFrame(everything, columns=self._column_names)
            return df
        else:
            J_min_i = np.argmin(Js)
            return xRs[J_min_i]

    def fit_instance(self, x_0: np.ndarray, verbose=False):
        x_0 = np.hstack((x_0, 1))
        x_r = self.runPSDInvScalingAllThetas(x_0, abstol=1e-7, n_epochs=2000, lr0=2.5,
                                              power_t=0.5, returnDataFrame=False)
        
        x_r = x_r[:-1]
        w = torch.from_numpy(self.weights.copy()).float()
        b = torch.tensor(self.bias).float()
        f_x = torch.sigmoid(torch.dot(torch.from_numpy(x_r.copy()).float(), w) + b).float()
        self.feasible = f_x.data.item() > 0.5
        return x_r