from ConformalPrediction.APS import AdaptivaPredictionSet


def create_conformal(conformal: str,
                     alpha: float,
                     randomized: bool,
                     allow_zero_sets: bool):
    """
    This function is to generate a conformal prediction class, constructing prediction sets.

    Args: 
        conformal: The type of conformal prediction method you want to create.
                   Available options: APS

        alpha: The user specified error rate, i.e., the prediction sets are guranteed to include the groud-true label 
               with probability 1-alpha
               eg. 0.1

        randomized: Whether to use randomize process. We recommand you to set True to achieve expected coverage rate.
                    Available options: True, False

        allow_zero_set: Whther to allow zero set. We recommand you to set True to achieve expected coverage rate and to reject 
                        suspicous predictions.
                        Available options: True, False

    Returns:
        A conformal prediction class.
    """

    if conformal == "APS":
        return AdaptivaPredictionSet(alpha, randomized, allow_zero_sets)

    else:
        return NotImplementedError("Please check whether there's a typo, or your method is not available!")