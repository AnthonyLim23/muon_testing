from quickBayes.workflow.model_template import ModelSelectionWorkflow
from quickBayes.functions.base import BaseFitFunction
from numpy import ndarray
import numpy as np
from typing import Dict, List
from quickBayes.functions.BG import (FlatBG,
                                     )

class LinearBG(BaseFitFunction):
    def __init__(self, prefix: str = ''):
        """
        :param prefix: prefix for function parameters in report
        """
        super().__init__(2, prefix, [0., 0], [-1., -1.], [1., 1.])

    @property
    def constant(self) -> str:
        return str(f'{self._prefix}BG constant')

    @property
    def grad(self) -> str:
        return str(f'{self._prefix}BG gradient')

        return str(f'{self._prefix}BG quad')

    def __call__(self, x: ndarray, m: float, c: float) -> ndarray:
        """
        Implement the Linear BG.
        Need to follow the expected
        form for scipy
        :param x: x values
        :param m: gradient
        :param c: constant
        :return linear background y values
        """
        xx = np.array(x)
        return m*xx + c

    def read_from_report(self, report_dict: Dict[str, List[float]],
                         index: int = 0) -> List[float]:
        """
        Read the parameters from the results dict
        :param report_dict: the dict of results
        :param index: the index to get results from
        :return the parameters
        """
        return [
                self._read_report(report_dict, self.grad, index),
                self._read_report(report_dict, self.constant, index)]

    def report(self, report_dict: Dict[str, List[float]],
               m: float, c: float) -> Dict[str, List[float]]:
        """
        reporting method
        :param report_dict: dict of parameters
        :param m: gradient
        :param c: constant
        :return dict of parameters, including BG
        """
        report_dict = self._add_to_report(self.grad,
                                          m, report_dict)
        report_dict = self._add_to_report(self.constant,
                                          c, report_dict)
        return report_dict



class QuadBG(BaseFitFunction):
    def __init__(self, prefix: str = ''):
        """
        :param prefix: prefix for function parameters in report
        """
        super().__init__(3, prefix, [1., 1., 1], [0., 0., 0.], [5., 5., 5.])

    @property
    def constant(self) -> str:
        return str(f'{self._prefix}BG constant')

    @property
    def grad(self) -> str:
        return str(f'{self._prefix}BG gradient')

    @property
    def quad(self) -> str:
        return str(f'{self._prefix}BG quad')

    def __call__(self, x: ndarray, b: float, m: float, c: float) -> ndarray:
        """
        Implement the Linear BG.
        Need to follow the expected
        form for scipy
        :param x: x values
        :param m: gradient
        :param c: constant
        :return linear background y values
        """
        xx = np.array(x)
        return m*xx + c + b*xx*xx

    def read_from_report(self, report_dict: Dict[str, List[float]],
                         index: int = 0) -> List[float]:
        """
        Read the parameters from the results dict
        :param report_dict: the dict of results
        :param index: the index to get results from
        :return the parameters
        """
        return [self._read_report(report_dict, self.quad, index),
                self._read_report(report_dict, self.grad, index),
                self._read_report(report_dict, self.constant, index)]

    def report(self, report_dict: Dict[str, List[float]],
               b: float, m: float, c: float) -> Dict[str, List[float]]:
        """
        reporting method
        :param report_dict: dict of parameters
        :param m: gradient
        :param c: constant
        :return dict of parameters, including BG
        """
        report_dict = self._add_to_report(self.quad,
                                          b, report_dict)
        report_dict = self._add_to_report(self.grad,
                                          m, report_dict)
        report_dict = self._add_to_report(self.constant,
                                          c, report_dict)
        return report_dict



class Test(ModelSelectionWorkflow):
    """
    A class for the muon exponential decay workflow
    """
    def preprocess_data(self, x_data: ndarray,
                        y_data: ndarray, e_data: ndarray,
                        ) -> None:
        """
        The preprocessing needed for the data.
        This crops and stores the data.
        :param x_data: the x data to fit to
        :param y_data: the y data to fit to
        :param e_data: the errors for the y data
        """
        super().preprocess_data(x, y, e)


    def update_function(self, func: BaseFitFunction, N: int) -> BaseFitFunction:
        function = self._update_function(N)
        function.update_prefix(f'N{N}:')
        return function


    def _update_function(self, N: int) -> BaseFitFunction:
        """
        This method adds a exponential decay to the fitting
        function.
        :param func: the fitting function that needs modifying
        :return the modified fitting function
        """

        if N == 1:
            print('flat', N)
            a = FlatBG()
            a.set_bounds([0], [5])
            a.set_guess([1])
            return a
        elif N == 2:
            print('linear', N)
            a = LinearBG()
            a.set_bounds([0, 0], [5,5])
            a.set_guess([1., 1])
            return a
        else:
            print('quad', N)
            return QuadBG()


x = np.linspace(0, 1.7, 30)


def generate_data(x, noise, alpha, epsilon):
    y = 2.67*np.ones(len(x)) + 4.05*x + alpha*x*x
    y *= 1 + epsilon*noise
    e = epsilon * y
    return y, e

np.random.seed(13)
noise = np.random.normal(0, 0.2, len(x))
y, e = generate_data(x, noise, 0.2, .1)


import matplotlib.pyplot as plt
from copy import deepcopy as cp

quad = []
linear = []
alpha = [0., 0.05, 0.1, 0.15, 0.2, 0.23, 0.24, 0.25, 0.26, 0.27, 0.3, 0.35, 0.4]

insert = False


fig, ax = plt.subplots()
plt.rcParams.update({'font.size':15})
ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

#plt.rc('axes', labelsize=20)
for ax2 in alpha:
    print(ax2)

    #plt.errorbar(x,y, e, label='data')
    y, e = generate_data(x, noise, ax2, 0.1)
    wf = Test({}, {})
    wf.preprocess_data(x, y, e)
    func = FlatBG()
    wf.set_scipy_engine([], [], [])
    wf.execute(3, None, [])
    # report results
    params, errors = wf.get_parameters_and_errors
    
    #ll = params[f"N{1}:loglikelihood"]
    #print(f'flat loglikelihood', ll)
    #fx = FlatBG()
    #plt.plot(x, fx(x, params['N1:BG constant'][0]), label=f'Flat: {ll}')
    
    linear.append(params[f"N{2}:loglikelihood"])
    #print(f'linear loglikelihood', ll)
    
    quad.append(params[f"N{3}:loglikelihood"])
    if quad[-1] >= linear[-1] and not insert:
        insert = True
        tmp = fig.add_axes([.2, .15, .35, .35])
        tmp.errorbar(x, y, e, marker='x', color='k', linestyle='',label=f'Data for $\\alpha={ax2:.3f}$')
        fx = LinearBG()
        tmp.plot(x, fx(x, params['N2:BG gradient'][0], params['N2:BG constant'][0]), color='b', label=f'Linear: {linear[-1][0]:.3f}')
        fx = QuadBG()
        tmp.plot(x, fx(x, params['N3:BG quad'], params['N3:BG gradient'][0], params['N3:BG constant'][0]),color='r', linestyle='--', label=f'Quadratic: {quad[-1][0]:.3f}')
        tmp.legend()


    #print(f'quad loglikelihood', ll)
    
#fig, ax = plt.subplots()
ax.plot(alpha, np.array(linear), 'bs-', label='Linear')
ax.plot(alpha, np.array(quad), 'ro-', label='Quadratic')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('Loglikelihood')
ax.grid(True, 'major')
ax.set_title('Fits to the equation $2.67 + 4.05x + \\alpha x^2$')
plt.rcParams.update({'font.size':20})
ax.legend()
plt.show()
#for key in errors.keys():
#    print(key, params[key], errors[key])

