import logging

import torch


def set_logging() -> None:
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')


class ExcitatorySynapticWeightEquation:
    class FirstTerm:
        def __init__(self, alpha_filter: torch.Tensor, epsilon_filter: torch.Tensor, no_filter: torch.Tensor,
                     f_prime_u_i: torch.Tensor, from_layer_most_recent_spike: torch.Tensor) -> None:
            self.alpha_filter = alpha_filter
            self.epsilon_filter = epsilon_filter
            self.no_filter = no_filter
            self.f_prime_u_i = f_prime_u_i
            self.prev_layer_most_recent_spike = from_layer_most_recent_spike

    class SecondTerm:
        def __init__(self, alpha_filter: torch.Tensor, no_filter: torch.Tensor, prediction_error: torch.Tensor,
                     deviation_scale: torch.Tensor, deviation: torch.Tensor) -> None:
            self.alpha_filter = alpha_filter
            self.no_filter = no_filter
            self.prediction_error = prediction_error
            self.deviation_scale = deviation_scale
            self.deviation = deviation

    def __init__(self, first_term: FirstTerm, second_term: SecondTerm) -> None:
        self.first_term = first_term
        self.second_term = second_term
