import logging
from typing import Optional, Self

import torch
from torch import nn
import wandb

from model.src.constants import DECAY_BETA, DELTA, KAPPA, LAMBDA_HEBBIAN, THETA_REST, XI, ZENKE_BETA
from model.src.logging_util import ExcitatorySynapticWeightEquation
from model.src.settings import LayerSettings
from model.src.util import ExcitatorySynapseFilterGroup, InhibitoryPlasticityTrace, MovingAverageLIF, SynapticUpdateType

PERCENTAGE_INHIBITORY = 50


def inhibitory_mask_vec(length: int, percentage_ones: int) -> torch.Tensor:
    num_ones = int(length * (percentage_ones / 100))
    vector = torch.zeros(length)
    indices = torch.randperm(length)[:num_ones]
    vector[indices] = 1
    return vector


class Layer(nn.Module):
    def __init__(self, layer_settings: LayerSettings) -> None:
        super().__init__()

        self.layer_settings = layer_settings

        # weights from prev layer to this layer
        self.forward_weights = nn.Linear(
            layer_settings.prev_size, layer_settings.size)
        torch.nn.init.uniform_(self.forward_weights.weight, a=0.1, b=1.0)

        # weights from next layer to this layer
        self.backward_weights = nn.Linear(layer_settings.next_size, layer_settings.size)
        torch.nn.init.uniform_(self.backward_weights.weight, a=0.1, b=1.0)

        # weights from this layer to this layer
        self.recurrent_weights = nn.Linear(layer_settings.size, layer_settings.size)
        torch.nn.init.uniform_(self.recurrent_weights.weight, a=0.1, b=1.0)

        self.inhibitory_mask_vec = inhibitory_mask_vec(
            layer_settings.size, PERCENTAGE_INHIBITORY)
        self.excitatory_mask_vec = (~self.inhibitory_mask_vec.bool()).int(
        ).float()
        self.register_buffer("inhibitory_mask_vec_", self.inhibitory_mask_vec)
        self.register_buffer("excitatory_mask_vec_", self.excitatory_mask_vec)
        # check that inhib is opposite of excit
        assert torch.all(self.inhibitory_mask_vec + self.excitatory_mask_vec == 1)

        self.lif = MovingAverageLIF(batch_size=layer_settings.batch_size, layer_size=layer_settings.size,
                                    beta=DECAY_BETA, device=layer_settings.device)

        self.prev_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None

        self.forward_filter_group = ExcitatorySynapseFilterGroup()
        self.recurrent_filter_group = ExcitatorySynapseFilterGroup()
        self.backward_filter_group = ExcitatorySynapseFilterGroup()
        self.filter_groups = nn.ModuleList([self.forward_filter_group, self.recurrent_filter_group,
                                           self.backward_filter_group])

        trace_shape = (layer_settings.batch_size, layer_settings.size)
        self.inhibitory_trace = InhibitoryPlasticityTrace(device=self.layer_settings.device, trace_shape=trace_shape)

        self.forward_counter = 0

    def set_next_layer(self, next_layer: Self) -> None:
        self.next_layer = next_layer

    def set_prev_layer(self, prev_layer: Self) -> None:
        self.prev_layer = prev_layer

    def forward(self, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        # NOTE: The current takes into account the bias from the `Linear` weights
        with torch.no_grad():
            excitatory_recurrent_mask = self.excitatory_mask_vec

            # recurrent
            inhib_recurrent_masked = self.inhibitory_mask_vec.unsqueeze(0).expand(
                self.layer_settings.size, -1) * self.recurrent_weights.weight
            excitatory_recurrent_masked = excitatory_recurrent_mask.unsqueeze(0).expand(
                self.layer_settings.size, -1) * self.recurrent_weights.weight

            assert inhib_recurrent_masked.shape == self.recurrent_weights.weight.shape
            assert excitatory_recurrent_masked.shape == self.recurrent_weights.weight.shape

            recurrent_input = self.lif.spike_moving_average.spike_rec[-1]
            recurrent_current_inhibitory = torch.nn.functional.linear(
                recurrent_input, inhib_recurrent_masked, self.recurrent_weights.bias)
            recurrent_current_excitatory = torch.nn.functional.linear(
                recurrent_input, excitatory_recurrent_masked, self.recurrent_weights.bias)
            recurrent_contribution = recurrent_current_excitatory - recurrent_current_inhibitory

            total_current = recurrent_contribution
            assert total_current.shape == (self.layer_settings.batch_size, self.layer_settings.size)

            # forward
            if data is not None:
                forward_contribution = self.forward_weights(data)
            else:
                assert self.prev_layer is not None

                excitatory_forward_mask = self.prev_layer.excitatory_mask_vec

                inhib_forward_masked = self.prev_layer.inhibitory_mask_vec.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.forward_weights.weight
                excitatory_forward_masked = excitatory_forward_mask.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.forward_weights.weight

                assert inhib_forward_masked.shape == self.forward_weights.weight.shape
                assert excitatory_forward_masked.shape == self.forward_weights.weight.shape

                forward_input = self.prev_layer.lif.spike_moving_average.spike_rec[-1]
                forward_current_inhibitory = torch.nn.functional.linear(
                    forward_input, inhib_forward_masked, self.forward_weights.bias)
                forward_current_excitatory = torch.nn.functional.linear(
                    forward_input, excitatory_forward_masked, self.forward_weights.bias)
                forward_contribution = forward_current_excitatory - forward_current_inhibitory

            total_current += forward_contribution

            # backward
            if self.next_layer is not None:
                excitatory_backward_mask = self.next_layer.excitatory_mask_vec

                inhib_backward_masked = self.next_layer.inhibitory_mask_vec.unsqueeze(0).expand(
                    self.layer_settings.size, -1) * self.backward_weights.weight
                excitatory_backward_masked = excitatory_backward_mask.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.backward_weights.weight

                assert inhib_backward_masked.shape == self.backward_weights.weight.shape
                assert excitatory_backward_masked.shape == self.backward_weights.weight.shape

                backward_input = self.next_layer.lif.spike_moving_average.spike_rec[-1]
                backward_current_inhibitory = torch.nn.functional.linear(
                    backward_input, inhib_backward_masked, self.backward_weights.bias)
                backward_current_excitatory = torch.nn.functional.linear(
                    backward_input, excitatory_backward_masked, self.backward_weights.bias)

                backward_contribution = backward_current_excitatory - backward_current_inhibitory
                total_current += backward_contribution

        # forward pass
        spk = self.lif.forward(total_current)
        self.forward_counter += 1

        logging.debug("")
        logging.debug(f"current: {str(total_current)}")
        logging.debug(f"mem: {str(self.lif.mem())}")
        logging.debug(f"spk: {str(spk)}")

        return spk

    # TODO: this will need to be removed or refactored once we move to more complex network topologies
    def __log_equation_context(self, excitatory_equation: ExcitatorySynapticWeightEquation, dw_dt: torch.Tensor,
                               spike: torch.Tensor, mem: torch.Tensor) -> None:
        logging.debug("")
        logging.debug("first term stats:")
        logging.debug(
            f"prev layer most recent spike: {excitatory_equation.first_term.prev_layer_most_recent_spike}")
        logging.debug(
            f"zenke beta * abs: {ZENKE_BETA * abs(mem - THETA_REST)}")
        logging.debug(
            f"f prime u i: {excitatory_equation.first_term.f_prime_u_i}")
        logging.debug(
            f"first term no filter: {excitatory_equation.first_term.no_filter}")
        logging.debug(
            f"first term epsilon: {excitatory_equation.first_term.epsilon_filter}")
        logging.debug(
            f"first term alpha filter: {excitatory_equation.first_term.alpha_filter}")
        logging.debug("")
        logging.debug("second term stats:")
        logging.debug(
            f"second_term_prediction_error: {excitatory_equation.second_term.prediction_error}")
        logging.debug(
            f"second_term_deviation_scale: {excitatory_equation.second_term.deviation_scale}")
        logging.debug(
            f"second_term_deviation: {excitatory_equation.second_term.deviation}")
        logging.debug(
            f"second term no filter: {excitatory_equation.second_term.no_filter}")
        logging.debug(
            f"second term alpha: {excitatory_equation.second_term.alpha_filter}")
        logging.debug("")
        logging.debug(
            f"first term alpha: {excitatory_equation.first_term.alpha_filter}")
        logging.debug(
            f"second term alpha: {excitatory_equation.second_term.alpha_filter}")
        logging.debug("")
        logging.debug(f"dw_dt shape: {dw_dt.shape}")
        logging.debug(f"dw_dt: {dw_dt}")
        logging.debug(
            f"forward weights shape: {self.forward_weights.weight.shape}")
        logging.debug(f"forward weights: {self.forward_weights.weight}")

        def reduce_feature_dims_with_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return tensor[mask.unsqueeze(0).expand(self.layer_settings.batch_size, -1).bool()]

        # NOTE: if we want to log just one data point index the batch dim into these tensors
        excitatory_mem = reduce_feature_dims_with_mask(mem, self.excitatory_mask_vec)
        inhibitory_mem = reduce_feature_dims_with_mask(mem, self.inhibitory_mask_vec)
        excitatory_spike = reduce_feature_dims_with_mask(spike, self.excitatory_mask_vec)
        inhibitory_spike = reduce_feature_dims_with_mask(spike, self.inhibitory_mask_vec)
        wandb.log(
            {f"layer_{self.layer_settings.layer_id}_exc_mem": excitatory_mem[0].mean()}, step=self.forward_counter)
        wandb.log(
            {f"layer_{self.layer_settings.layer_id}_inh_mem": inhibitory_mem[0].mean()}, step=self.forward_counter)
        wandb.log({f"layer_{self.layer_settings.layer_id}_exc_spike": excitatory_spike[0].mean()},
                  step=self.forward_counter)
        wandb.log({f"layer_{self.layer_settings.layer_id}_inh_spike": inhibitory_spike[0].mean()},
                  step=self.forward_counter)
        wandb.log({f"layer_{self.layer_settings.layer_id}_data_point_0": self.data[0][0]}, step=self.forward_counter)
        wandb.log({f"layer_{self.layer_settings.layer_id}_data_point_1": self.data[0][1]}, step=self.forward_counter)

        # TODO: The below metrics are specific to the dataset so will eventually need to be removed

        if self.layer_settings.layer_id == 0:
            # Log for a layer the weight from the first datapoint to the excitatory
            # neuron. The key here is that we need to know what the excitatory
            # neuron is in order to figure out how to index into the forward
            # weights.
            excitatory_masked_weight = self.excitatory_mask_vec.unsqueeze(1).expand(-1, self.layer_settings.data_size) \
                * self.forward_weights.weight
            # Identify rows that are not all zeros
            non_zero_rows = excitatory_masked_weight.any(dim=1)
            # Filter out rows that are all zeros
            excitatory_masked_weight = excitatory_masked_weight[non_zero_rows]
            assert excitatory_masked_weight.shape == (self.layer_settings.size / 2, self.layer_settings.data_size)

            wandb.log({f"layer_{self.layer_settings.layer_id}_exc_weight_0": excitatory_masked_weight[0][0]},
                      step=self.forward_counter)
            wandb.log(
                {f"layer_{self.layer_settings.layer_id}_exc_weight_1": excitatory_masked_weight[0][1]},
                step=self.forward_counter)

    def train_excitatory_from_layer(self, synaptic_update_type: SynapticUpdateType, spike: torch.Tensor,
                                    filter_group: ExcitatorySynapseFilterGroup, from_layer: Optional[Self],
                                    data: torch.Tensor) -> None:
        """
        The LPL excitatory learning rule is implemented here. It is defined as dw_ji/dt,
        for which we optimize the computation with matrices.

        This learning rule is broken up into three terms:

         1. The first term contains S_j(t) * f'(U_i(t)). We form a matrix via
            some unsqueezes to form a first term matrix of size (batch_size, i,
            j). This is performed with outer product.

         2. The second term contains S_i. We form a matrix via some unsqueezes
            to form a second term matrix of size (batch_size, i, j). This is
            performed by expanding and copying the tensor along the j dimension.

        The final dw_ij/dt is formed by a Hadamard product of the first term and
        the second term. This is then summed across the batch dimension and
        divided by the batch size to form the final dw_ij/dt matrix. We then
        mask this to filter out the inhibitory weights and apply this to the
        excitatory weights.
        """

        # NOTE: The only time `from_layer` is None is if the first layer is
        # training its forward weights. In this case `data` will drive the
        # update.
        from_layer_size = from_layer.layer_settings.size if from_layer is not None else self.layer_settings.data_size

        if from_layer is None:
            mask = torch.ones(self.layer_settings.size, self.layer_settings.data_size).to(self.layer_settings.device)
        else:
            # flip the mask as this is for excitatory connections
            mask = (~from_layer.inhibitory_mask_vec.bool()).int()
            # expand the mask across the synaptic weight matrix
            mask = mask.unsqueeze(0).expand(self.layer_settings.size, -1)
            assert mask.shape == (self.layer_settings.size, from_layer_size)

        with torch.no_grad():
            # first term
            f_prime_u_i = ZENKE_BETA * \
                (1 + ZENKE_BETA * abs(self.lif.mem() - THETA_REST)) ** (-2)
            f_prime_u_i = f_prime_u_i.unsqueeze(2)
            from_layer_most_recent_spike: torch.Tensor = from_layer.lif.spike_moving_average.spike_rec[
                0] if from_layer is not None else data  # type: ignore [union-attr, assignment]
            from_layer_most_recent_spike = from_layer_most_recent_spike.unsqueeze(
                1)
            first_term_no_filter = f_prime_u_i @ from_layer_most_recent_spike
            first_term_epsilon = filter_group.first_term_epsilon.apply(
                first_term_no_filter)
            first_term_alpha = filter_group.first_term_alpha.apply(
                first_term_epsilon)

            # assert shapes
            assert f_prime_u_i.shape == (self.layer_settings.batch_size, self.layer_settings.size, 1)
            assert from_layer_most_recent_spike.shape == (self.layer_settings.batch_size, 1, from_layer_size)
            assert first_term_no_filter.shape == (self.layer_settings.batch_size,
                                                  self.layer_settings.size, from_layer_size)
            assert first_term_epsilon.shape == (self.layer_settings.batch_size,
                                                self.layer_settings.size, from_layer_size)
            assert first_term_alpha.shape == (self.layer_settings.batch_size,
                                              self.layer_settings.size, from_layer_size)

            # second term
            current_layer_most_recent_spike = self.lif.spike_moving_average.spike_rec[
                0]
            current_layer_delta_t_spikes_ago = self.lif.spike_moving_average.spike_rec[-1]
            current_layer_spike_moving_average = self.lif.spike_moving_average.tracked_value()
            current_layer_variance_moving_average = self.lif.variance_moving_average.tracked_value()

            second_term_prediction_error = current_layer_most_recent_spike - \
                current_layer_delta_t_spikes_ago
            second_term_deviation_scale = LAMBDA_HEBBIAN / \
                (current_layer_variance_moving_average + XI)
            second_term_deviation = current_layer_most_recent_spike - \
                current_layer_spike_moving_average
            second_term_no_filter = -1 * \
                (second_term_prediction_error) + second_term_deviation_scale * \
                second_term_deviation + DELTA
            # this can potentially be done after the filter
            second_term_no_filter = second_term_no_filter.unsqueeze(2).expand(-1, -1, from_layer_size)
            second_term_alpha = filter_group.second_term_alpha.apply(
                second_term_no_filter)

            # assert shapes
            assert second_term_deviation.shape == (self.layer_settings.batch_size, self.layer_settings.size)
            assert second_term_no_filter.shape == (self.layer_settings.batch_size,
                                                   self.layer_settings.size, from_layer_size)
            assert second_term_alpha.shape == (self.layer_settings.batch_size,
                                               self.layer_settings.size, from_layer_size)

            # update weights
            dw_dt = self.layer_settings.learning_rate * (first_term_alpha *
                                                         second_term_alpha)
            dw_dt = dw_dt.sum(0) / dw_dt.shape[0] * mask

            match synaptic_update_type:
                case SynapticUpdateType.RECURRENT:
                    weight_ref = self.recurrent_weights
                case SynapticUpdateType.FORWARD:
                    weight_ref = self.forward_weights
                case SynapticUpdateType.BACKWARD:
                    weight_ref = self.backward_weights
                case _:
                    raise ValueError("Invalid synaptic update type")

            weight_ref.weight += dw_dt

            # only log for first layer and forward connections
            # TODO: remove or refactor when learning rule is stable
            if self.prev_layer is None and synaptic_update_type == SynapticUpdateType.FORWARD:
                first_term = ExcitatorySynapticWeightEquation.FirstTerm(
                    alpha_filter=first_term_alpha,
                    epsilon_filter=first_term_epsilon,
                    no_filter=first_term_no_filter,
                    f_prime_u_i=f_prime_u_i,
                    from_layer_most_recent_spike=from_layer_most_recent_spike
                )
                second_term = ExcitatorySynapticWeightEquation.SecondTerm(
                    alpha_filter=second_term_alpha,
                    no_filter=second_term_no_filter,
                    prediction_error=second_term_prediction_error,
                    deviation_scale=second_term_deviation_scale,
                    deviation=second_term_deviation
                )
                synaptic_weight_equation = ExcitatorySynapticWeightEquation(
                    first_term=first_term,
                    second_term=second_term,
                )

                # TODO: Remove this when we decouple the logging for the
                # pointcloud benchmark from the model code
                self.data = data
                self.__log_equation_context(synaptic_weight_equation, dw_dt, spike, self.lif.mem())

    def train_inhibitory_from_layer(self, synaptic_update_type: SynapticUpdateType, spike: torch.Tensor,
                                    from_layer: Self) -> None:
        # expand the mask across the synaptic weight matrix
        mask = from_layer.inhibitory_mask_vec.unsqueeze(0).expand(self.layer_settings.size, -1)
        assert mask.shape == (self.layer_settings.size, from_layer.layer_settings.size)

        self.inhibitory_trace.apply(spike)

        with torch.no_grad():
            x_i = self.inhibitory_trace.tracked_value().unsqueeze(2).expand(-1, -1, from_layer.layer_settings.size)
            S_j = from_layer.lif.spike_moving_average.spike_rec[-1].unsqueeze(
                1).expand(-1, self.layer_settings.size, -1)
            assert x_i.shape == S_j.shape

            # x_i * S_j
            first_term = (x_i - 2 * KAPPA *
                          self.inhibitory_trace.tau_stdp) * S_j
            assert first_term.shape == (self.layer_settings.batch_size,
                                        self.layer_settings.size, from_layer.layer_settings.size)

            # S_i * x_j
            S_i = self.lif.spike_moving_average.spike_rec[-1].unsqueeze(
                2).expand(-1, -1, from_layer.layer_settings.size)
            x_j = from_layer.inhibitory_trace.tracked_value().unsqueeze(1).expand(
                -1, self.layer_settings.size, -1)

            second_term = S_i * x_j
            assert second_term.shape == (self.layer_settings.batch_size,
                                         self.layer_settings.size, from_layer.layer_settings.size)

            dw_dt = self.layer_settings.learning_rate * (first_term + second_term)
            dw_dt = dw_dt.sum(0) / dw_dt.shape[0]

            # update weights and apply mask
            dw_dt = dw_dt * mask

            # update weights
            match synaptic_update_type:
                case SynapticUpdateType.RECURRENT:
                    weight_ref = self.recurrent_weights
                case SynapticUpdateType.FORWARD:
                    weight_ref = self.forward_weights
                case SynapticUpdateType.BACKWARD:
                    weight_ref = self.backward_weights
                case _:
                    raise ValueError("Invalid synaptic update type")

            weight_ref.weight += dw_dt

    # TODO: uncomment when we move to more complex network topologies
    def train_synapses(self, spike: torch.Tensor, data: torch.Tensor) -> None:
        # recurrent connections always trained
        self.train_excitatory_from_layer(SynapticUpdateType.RECURRENT, spike, self.recurrent_filter_group, self, data)
        self.train_inhibitory_from_layer(SynapticUpdateType.RECURRENT, spike, self)

        if self.next_layer is not None:
            self.train_excitatory_from_layer(
                SynapticUpdateType.BACKWARD, spike, self.backward_filter_group, self.next_layer, data)
            self.train_inhibitory_from_layer(SynapticUpdateType.BACKWARD, spike, self.next_layer)

        # if prev layer is None then forward connections driven by data
        self.train_excitatory_from_layer(SynapticUpdateType.FORWARD, spike,
                                         self.forward_filter_group, self.prev_layer, data)

        # no forward connections from data are treated as inhibitory
        if self.prev_layer is not None:
            self.train_inhibitory_from_layer(SynapticUpdateType.FORWARD, spike, self.prev_layer)

        logging.info(f"trained layer {self.layer_settings.layer_id} synapses")
