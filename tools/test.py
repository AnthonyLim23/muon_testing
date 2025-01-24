
import numpy as np
from quickBayes.fit_functions.gaussian import Gaussian
from quickBayes.functions.composite import CompositeFunction
from quickBayes.utils.crop_data import crop
from quickBayes.workflow.model_template import ModelSelectionWorkflow
from quickBayes.functions.base import BaseFitFunction
from numpy import ndarray
from typing import Dict, List


class GaussianExample(ModelSelectionWorkflow):
    """
    A class for the finding gaussians
    """
    def preprocess_data(self, x_data: ndarray,
                        y_data: ndarray, e_data: ndarray,
                        start_x: float, end_x: float) -> None:
        """
        The preprocessing needed for the data.
        This crops and stores the data.
        :param x_data: the x data to fit to
        :param y_data: the y data to fit to
        :param e_data: the errors for the y data
        :param start_x: the start x value
        :param end_x: the end x value
        """
        sx, sy, se = crop(x_data, y_data, e_data,
                          start_x, end_x)
        super().preprocess_data(sx, sy, se)

    @staticmethod
    def _update_function(func: BaseFitFunction) -> BaseFitFunction:
        """
        This method adds a Gaussian to the fitting
        function.
        :param func: the fitting function that needs modifying
        :return the modified fitting function
        """

        g_function = Gaussian()
        # need to change the bounds and guess on the function
        g_function.set_bounds([80, 0, .2], [120, 10, 2])
        g_function.set_guess([100, 5, 1.2])
        func.add_function(g_function)
        return func


# generate some data
x = np.linspace(-50, 50, 1000)
noise = 1 + 0.1 * (np.random.normal(0, .2, len(x)))
gauss = Gaussian()
ground_truth = gauss (x, 103, 4.2, .9)
y = ground_truth * noise
e = np.power(y, 0.5) # normal for count data

# setup 
params = {}
errors = {}
func = CompositeFunction() # could add a background in needed
workflow = GaussianExample(params, errors)
workflow.preprocess_data(x, y, e, -10., 10.)
workflow.set_scipy_engine([], [], [])
# this will crop the data
# execute the workflow with no intial guess and up to 5 gaussians
_ = workflow.execute(max_num_features=5, func=func, params=[])

# report results
params, errors = workflow.get_parameters_and_errors
print(params.keys())
for j in range(5):

    print(f"{j + 1} peak(s)")
    print('loglikelihood', params[f"N{j+1}:loglikelihood"])
    for key in errors.keys():
        if f"N{j+1}" in key:
            print(key, params[key], errors[key])
    
