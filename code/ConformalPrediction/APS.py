import numpy as np
import torch

from ConformalPrediction.conformal import ConformalPrediction
from Utils.utils import sort_sum


class AdaptivaPredictionSet(ConformalPrediction):

    def __init__(self, alpha, randomized, allow_zero_sets) -> None:
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        super().__init__(alpha)


    def forward(self, X):
        return super().forward(X)
    

    def solve_qhat(self, calib_probs_loader):
        super().solve_qhat(calib_probs_loader)


    def solve_nonconformity_score(self, calib_probs_loader):

        with torch.no_grad():

            score_random = []

            for probs, targets in calib_probs_loader:
                probs = probs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                I, ordered, cumsum = sort_sum(probs)
                target_order = np.array([np.where(row == target_value)[0][0] for row, target_value in zip(I, targets)])
                score_no_random = cumsum[np.arange(len(cumsum)), target_order]
                target_proba = probs[(np.arange(probs.shape[0]), targets)]

                for i in range(probs.shape[0]):
                    u = np.random.random()
                    score_random.append(u*target_proba[i] + score_no_random[i] - target_proba[i])

            score_random = np.array(score_random)
            self.scores = score_random

        return score_random
    

    def solve_prediction_sets(self, probs):

        with torch.no_grad():
            probs = probs.detach().cpu().numpy()

            I, ordered, cumsum = sort_sum(probs)
            q_hat = self.q_hat

            pred_set_size = (cumsum <= q_hat).sum(axis=1) + 1
            pred_set_size = np.minimum(pred_set_size, probs.shape[1])

            # If randomizing process is required
            if self.randomized:
                indices = np.arange(pred_set_size.shape[0])
                V = 1 / ordered[indices, pred_set_size - 1] * \
                    (q_hat - (cumsum[indices, pred_set_size - 1] - ordered[indices, pred_set_size - 1]))

                pred_set_size = pred_set_size - (np.random.random(V.shape) >= V).astype(int)

            # If q_hat = 1
            if q_hat == 1.0:
                pred_set_size[:] = cumsum.shape[1]
            
            # If zero set is not allowed
            if not self.allow_zero_sets:
                pred_set_size[pred_set_size==0] = 1

            # generating prediction sets
            pred_sets = []
            indices = np.arange(I.shape[0])
            pred_sets = [[I[i, 0:pred_set_size[i]],] for i in range(I.shape[0])]

        return pred_sets