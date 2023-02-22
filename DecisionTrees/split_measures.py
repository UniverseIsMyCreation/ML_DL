import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    sample = np.array(sample)
    sample = np.unique(sample, return_counts=True)
    frequency = sample[1] / sample[1].sum()
    error_metric = (sample[1].sum() - sample[1].max()) / sample[1].sum()
    gini_metric = sum(frequency * (1 - frequency))
    entropy_metric = -(sum(frequency * np.log(frequency)))
    measures = {'gini': float(gini_metric), 'entropy': float(entropy_metric), 'error': float(error_metric)}
    return measures
