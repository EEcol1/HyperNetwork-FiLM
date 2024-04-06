# Reference:
# https://github.com/DLR-RM/stable-baselines3/blob/8b3723c6d8420bb978f4d68409ff5189f87fe107/stable_baselines3/common/running_mean_std.py#L6

from typing import Tuple
import numpy as np

class RunningMeanStd():
    def __init__(self, num_tasks: int = 1, shape: Tuple[int, ...] = (), epsilon: float = 1e-4, ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.num_tasks = num_tasks
        shape = (num_tasks, ) + shape
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = np.zeros((num_tasks, ), np.float64) + epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, task_idx: int, arr: np.ndarray) -> None:
        if self.num_tasks == 1:
            task_idx = 0
        
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(task_idx, batch_mean, batch_var, batch_count)

    def update_from_moments(self, task_idx: int, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        if self.num_tasks == 1:
            task_idx = 0
        
        delta = batch_mean - self.mean[task_idx]
        tot_count = self.count[task_idx] + batch_count

        new_mean = self.mean[task_idx] + delta * batch_count / tot_count
        m_a = self.var[task_idx] * self.count[task_idx]
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count[task_idx] * batch_count / (self.count[task_idx] + batch_count)
        new_var = m_2 / (self.count[task_idx] + batch_count)

        new_count = batch_count + self.count[task_idx]

        self.mean[task_idx] = new_mean
        self.var[task_idx] = new_var
        self.count[task_idx] = new_count
    
    def normalize(self, task_idx, x):
        if self.num_tasks == 1:
            task_idx = 0
        return (x - self.mean[task_idx]) / np.sqrt(self.var[task_idx] + 1e-8)
    
    def normalize_batch(self, task_idx, x):
        if self.num_tasks == 1:
            task_idx = 0
        return (x - self.mean[task_idx][None, ...]) / np.sqrt(self.var[task_idx][None, ...] + 1e-8)
    
    def state_dict(self):
        out = {}
        for name in ["mean", "var", "count"]:
            out[name] = getattr(self, name)
        return out
    
    def load_state_dict(self, state_dict):
        for name in ["mean", "var", "count"]:
            setattr(self, name, state_dict[name])