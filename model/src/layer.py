import logging
from typing import Optional, Self

import numpy as np
import torch
from torch import nn
import wandb

from model.src.constants import DECAY_BETA, DELTA, EXC_TO_INHIB_CONN_C, EXC_TO_INHIB_CONN_SIGMA_SQUARED, \
    KAPPA, LAMBDA_HEBBIAN, LAYER_SPARSITY, PERCENTAGE_INHIBITORY, THETA_REST, XI, ZENKE_BETA
from model.src.logging_util import ExcitatorySynapticWeightEquation
from model.src.settings import LayerSettings
from model.src.util import ExcitatorySynapseFilterGroup, InhibitoryPlasticityTrace, MovingAverageLIF, SynapticUpdateType


def inhibitory_mask_vec(length: int, percentage_ones: int) -> torch.Tensor:
    num_ones = int(length * (percentage_ones / 100))
    vector = torch.zeros(length)
    indices = torch.randperm(length)[:num_ones]
    vector[indices] = 1
    return vector


class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_settings: LayerSettings,
                 synaptic_update_type: SynapticUpdateType, sparsity: float, bias: bool = False):
        """
        Initialize the SparseLinear module.

        Parameters:
        - in_features: size of each input sample
        - out_features: size of each output sample
        - bias: If set to False, the layer will not learn an additive bias. Default: False
        - sparsity: The fraction of weights to be set to zero. Default: 0.9
        """
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
        self.sparsity = sparsity
        self.mask: Optional[torch.Tensor] = None
        self.layer_settings = layer_settings
        self.synaptic_update_type = synaptic_update_type

    def set_sparsity_mask(self, from_layer: Optional['Layer'],
                          to_layer: Optional['Layer']) -> None:
        """
        Create a sparsity mask for the weights of the linear layer.
        """
        # create sparse mask
        mask = np.random.rand(
            self.out_features,
            self.in_features) > self.sparsity
        mask = torch.tensor(
            mask,
            dtype=torch.float32,
            requires_grad=False).to(
            self.layer_settings.device)

        if self.layer_settings.layer_id == 0 and self.synaptic_update_type == SynapticUpdateType.FORWARD:
            self.mask = mask
            return

        assert from_layer is not None
        assert to_layer is not None

        # don't apply sparse mask to exc_to_inh
        from_exc_mask = from_layer.excitatory_mask_vec
        to_inh_mask = to_layer.inhibitory_mask_vec
        exc_mask_full = from_exc_mask.unsqueeze(
            0).expand(to_inh_mask.shape[0], -1)
        inh_mask_full = to_inh_mask.unsqueeze(
            1).expand(-1, from_exc_mask.shape[0])
        exc_to_inhib_mask = exc_mask_full * inh_mask_full
        mask[exc_to_inhib_mask.bool()] = 0

        # apply exc_to_inh mask
        gaussian_mask = self.__gaussian_connection_probability_matrix() * \
            exc_to_inhib_mask
        exc_to_inhib_mask_post_gaussian = torch.bernoulli(gaussian_mask)
        # assert that the gaussian mask only has values filled in for where the
        # existing `mask` is 0
        assert torch.all(((mask != 0) * exc_to_inhib_mask_post_gaussian) == 0)
        mask = mask + exc_to_inhib_mask_post_gaussian

        self.mask = mask

    def __gaussian_connection_probability_matrix(
            self) -> torch.Tensor:
        indices_i = torch.arange(self.out_features, dtype=torch.float32).unsqueeze(
            1).expand(-1, self.in_features).to(self.layer_settings.device)
        indices_j = torch.arange(
            self.in_features, dtype=torch.float32).unsqueeze(0).expand(
            self.out_features, -1).to(self.layer_settings.device)

        pre_exp = -1 * (indices_j - self.layer_settings.exc_to_inhib_conn_c * (indices_i +
                        indices_j / self.layer_settings.exc_to_inhib_conn_c - indices_j)) ** 2 / self.layer_settings.exc_to_inhib_conn_sigma_squared
        return torch.exp(pre_exp)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SparseLinear module. Applies the mask to the weights
        before performing the linear operation.
        """
        # Apply the mask to the weights
        self.linear.weight.data *= self.mask
        out: torch.Tensor = self.linear(input)
        return out

    def weight(self) -> torch.Tensor:
        """
        Return the effective weights of the linear layer after applying the sparsity mask.
        """
        return self.linear.weight.data * self.mask


class Layer(nn.Module):
    def __init__(self, layer_settings: LayerSettings) -> None:
        super().__init__()

        self.layer_settings = layer_settings

        # weights from prev layer to this layer
        if layer_settings.layer_id == 0:
            self.forward_weights = SparseLinear(
                layer_settings.prev_size, layer_settings.size, layer_settings=layer_settings,
                sparsity=0, synaptic_update_type=SynapticUpdateType.FORWARD)
        else:
            self.forward_weights = SparseLinear(
                layer_settings.prev_size, layer_settings.size, layer_settings=layer_settings,
                sparsity=layer_settings.layer_sparsity, synaptic_update_type=SynapticUpdateType.FORWARD)

        # TODO: for the last layer these will be of size 0 which can probably be
        # refactored to handle this cleaner.
        #
        # weights from next layer to this layer
        self.backward_weights = SparseLinear(layer_settings.next_size, layer_settings.size,
                                             layer_settings=layer_settings,
                                             sparsity=layer_settings.layer_sparsity,
                                             synaptic_update_type=SynapticUpdateType.BACKWARD)

        # weights from this layer to this layer
        self.recurrent_weights = SparseLinear(layer_settings.size, layer_settings.size,
                                              layer_settings=layer_settings,
                                              sparsity=layer_settings.layer_sparsity,
                                              synaptic_update_type=SynapticUpdateType.RECURRENT)

        self.inhibitory_mask_vec_ = inhibitory_mask_vec(
            layer_settings.size, layer_settings.percentage_inhibitory).to(layer_settings.device)
        self.excitatory_mask_vec_ = (~self.inhibitory_mask_vec_.bool()).int(
        ).float().to(layer_settings.device)
        self.register_buffer("inhibitory_mask_vec", self.inhibitory_mask_vec_)
        self.register_buffer("excitatory_mask_vec", self.excitatory_mask_vec_)
        self.excitatory_mask_vec: torch.Tensor = self.excitatory_mask_vec
        self.inhibitory_mask_vec: torch.Tensor = self.inhibitory_mask_vec
        assert torch.all(
            self.inhibitory_mask_vec +
            self.excitatory_mask_vec == 1)

        self.lif = MovingAverageLIF(batch_size=layer_settings.batch_size, layer_size=layer_settings.size,
                                    beta=DECAY_BETA, dt=layer_settings.dt, device=self.layer_settings.device)

        self.prev_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None

        self.forward_filter_group = ExcitatorySynapseFilterGroup(
            self.layer_settings.device)
        self.recurrent_filter_group = ExcitatorySynapseFilterGroup(
            self.layer_settings.device)
        self.backward_filter_group = ExcitatorySynapseFilterGroup(
            self.layer_settings.device)

        trace_shape = (layer_settings.batch_size, layer_settings.size)
        self.inhibitory_trace = InhibitoryPlasticityTrace(
            device=self.layer_settings.device, trace_shape=trace_shape)

        self.forward_counter = 0

    def _apply(self, fn):  # type: ignore
        """
        Override apply, but we don't want to apply to sibling layers as that
        will cause a stack overflow. The hidden layers are contained in a
        collection in the higher-level RecurrentFFNet. They will all get the
        apply call from there.
        """
        # Remove `previous_layer` and `next_layer` temporarily
        previous_layer = self.prev_layer
        next_layer = self.next_layer
        self.prev_layer = None
        self.next_layer = None

        # Apply `fn` to each parameter and buffer of this layer
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        # Apply `fn` to submodules
        for module in self.children():
            module._apply(fn)

        # Restore `previous_layer` and `next_layer`
        self.prev_layer = previous_layer
        self.next_layer = next_layer

        return self

    def retreive_activations(self) -> torch.Tensor:
        return self.lif.spike_moving_average.spike_rec[-1]

    def set_next_layer(self, next_layer: Self) -> None:
        self.next_layer = next_layer

    def set_prev_layer(self, prev_layer: Self) -> None:
        self.prev_layer = prev_layer

    def set_sparsity_masks(self) -> None:
        """
        Set the sparsity masks for the weights of the linear layers.
        """
        self.forward_weights.set_sparsity_mask(self.prev_layer, self)

        if self.next_layer is not None:
            self.backward_weights.set_sparsity_mask(self.next_layer, self)

        self.recurrent_weights.set_sparsity_mask(self, self)

    def forward(self, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            excitatory_recurrent_mask = self.excitatory_mask_vec

            # recurrent
            inhib_recurrent_masked = self.inhibitory_mask_vec.unsqueeze(0).expand(
                self.layer_settings.size, -1) * self.recurrent_weights.weight()
            excitatory_recurrent_masked = excitatory_recurrent_mask.unsqueeze(0).expand(
                self.layer_settings.size, -1) * self.recurrent_weights.weight()

            assert inhib_recurrent_masked.shape == self.recurrent_weights.weight().shape
            assert excitatory_recurrent_masked.shape == self.recurrent_weights.weight().shape

            recurrent_input = self.lif.spike_moving_average.spike_rec[-1]
            recurrent_current_inhibitory = torch.nn.functional.linear(
                recurrent_input, inhib_recurrent_masked)
            recurrent_current_excitatory = torch.nn.functional.linear(
                recurrent_input, excitatory_recurrent_masked)
            recurrent_contribution = recurrent_current_excitatory - recurrent_current_inhibitory

            total_current = recurrent_contribution.detach().clone()
            assert total_current.shape == (
                self.layer_settings.batch_size,
                self.layer_settings.size)

            # forward
            if data is not None:
                forward_contribution = self.forward_weights(data)
            else:
                assert self.prev_layer is not None

                excitatory_forward_mask = self.prev_layer.excitatory_mask_vec

                inhib_forward_masked = self.prev_layer.inhibitory_mask_vec.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.forward_weights.weight()
                excitatory_forward_masked = excitatory_forward_mask.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.forward_weights.weight()

                assert inhib_forward_masked.shape == self.forward_weights.weight().shape
                assert excitatory_forward_masked.shape == self.forward_weights.weight().shape

                forward_input = self.prev_layer.lif.spike_moving_average.spike_rec[-1]
                forward_current_inhibitory = torch.nn.functional.linear(
                    forward_input, inhib_forward_masked)
                forward_current_excitatory = torch.nn.functional.linear(
                    forward_input, excitatory_forward_masked)
                forward_contribution = forward_current_excitatory - forward_current_inhibitory

            total_current += forward_contribution

            # backward
            if self.next_layer is not None:
                excitatory_backward_mask = self.next_layer.excitatory_mask_vec

                inhib_backward_masked = self.next_layer.inhibitory_mask_vec.unsqueeze(0).expand(
                    self.layer_settings.size, -1) * self.backward_weights.weight()
                excitatory_backward_masked = excitatory_backward_mask.unsqueeze(
                    0).expand(self.layer_settings.size, -1) * self.backward_weights.weight()

                assert inhib_backward_masked.shape == self.backward_weights.weight().shape
                assert excitatory_backward_masked.shape == self.backward_weights.weight().shape

                backward_input = self.next_layer.lif.spike_moving_average.spike_rec[-1]
                backward_current_inhibitory = torch.nn.functional.linear(
                    backward_input, inhib_backward_masked)
                backward_current_excitatory = torch.nn.functional.linear(
                    backward_input, excitatory_backward_masked)

                backward_contribution = backward_current_excitatory - backward_current_inhibitory
                total_current += backward_contribution

        assert total_current.shape == (
            self.layer_settings.batch_size,
            self.layer_settings.size)

        # forward pass
        spk = self.lif.forward(total_current)
        self.forward_counter += 1

        logging.debug("")
        logging.debug(f"current: {str(total_current)}")
        logging.debug(f"mem: {str(self.lif.mem())}")
        logging.debug(f"spk: {str(spk)}")

        return spk

    # TODO: this will need to be removed or refactored once we move to more
    # complex network topologies
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
            f"forward weights shape: {self.forward_weights.weight().shape}")
        logging.debug(f"forward weights: {self.forward_weights.weight()}")

        def reduce_feature_dims_with_mask(
                tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return tensor[mask.unsqueeze(0).expand(
                self.layer_settings.batch_size, -1).bool()]

        # NOTE: if we want to log just one data point index the batch dim into
        # these tensors
        excitatory_mem = reduce_feature_dims_with_mask(
            mem, self.excitatory_mask_vec)
        inhibitory_mem = reduce_feature_dims_with_mask(
            mem, self.inhibitory_mask_vec)
        excitatory_spike = reduce_feature_dims_with_mask(
            spike, self.excitatory_mask_vec)
        inhibitory_spike = reduce_feature_dims_with_mask(
            spike, self.inhibitory_mask_vec)

        wandb.log(
            {f"layer_{self.layer_settings.layer_id}_exc_mem": excitatory_mem[0].mean()}, step=self.forward_counter)
        try:
            wandb.log(
                {f"layer_{self.layer_settings.layer_id}_inh_mem": inhibitory_mem[0].mean()}, step=self.forward_counter)
            wandb.log({f"layer_{self.layer_settings.layer_id}_inh_spike": inhibitory_spike[0].mean()},
                      step=self.forward_counter)
        except IndexError:
            pass
        wandb.log({f"layer_{self.layer_settings.layer_id}_exc_spike": excitatory_spike[0].mean()},
                  step=self.forward_counter)
        wandb.log(
            {
                f"layer_{self.layer_settings.layer_id}_data_point_0": self.data[0][0]},
            step=self.forward_counter)
        wandb.log(
            {
                f"layer_{self.layer_settings.layer_id}_data_point_1": self.data[0][1]},
            step=self.forward_counter)

        # TODO: The below metrics are specific to the dataset so will eventually
        # need to be removed. For now we comment them out.
        # if self.layer_settings.layer_id == 0:
        #     # Log for a layer the weight from the first datapoint to the excitatory
        #     # neuron. The key here is that we need to know what the excitatory
        #     # neuron is in order to figure out how to index into the forward
        #     # weights.
        #     excitatory_masked_weight = self.excitatory_mask_vec.unsqueeze(1) \
        #         .expand(-1, self.layer_settings.data_size) \
        #         * self.forward_weights.weight()
        #     # Identify rows that are not all zeros
        #     non_zero_rows = excitatory_masked_weight.any(dim=1)
        #     # Filter out rows that are all zeros
        #     excitatory_masked_weight = excitatory_masked_weight[non_zero_rows]
        #     assert excitatory_masked_weight.shape == (
        #         self.layer_settings.size / 2, self.layer_settings.data_size)

        #     wandb.log({f"layer_{self.layer_settings.layer_id}_exc_weight_0": excitatory_masked_weight[0][0]},
        #               step=self.forward_counter)
        #     wandb.log(
        #         {f"layer_{self.layer_settings.layer_id}_exc_weight_1":
        #             excitatory_masked_weight[0][1]},
        #         step=self.forward_counter)

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
            mask = torch.ones(self.layer_settings.size, self.layer_settings.data_size).to(
                device=self.layer_settings.device)
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
                first_term_no_filter, self.layer_settings.dt)
            first_term_alpha = filter_group.first_term_alpha.apply(
                first_term_epsilon, self.layer_settings.dt)

            # assert shapes
            assert f_prime_u_i.shape == (
                self.layer_settings.batch_size, self.layer_settings.size, 1)
            assert from_layer_most_recent_spike.shape == (
                self.layer_settings.batch_size, 1, from_layer_size)
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
            second_term_no_filter = second_term_no_filter.unsqueeze(
                2).expand(-1, -1, from_layer_size)
            second_term_alpha = filter_group.second_term_alpha.apply(
                second_term_no_filter, self.layer_settings.dt)

            # assert shapes
            assert second_term_deviation.shape == (
                self.layer_settings.batch_size, self.layer_settings.size)
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

            weight_ref.linear.weight += dw_dt
            clamped_weights = torch.clamp(weight_ref.linear.weight, min=0)
            weight_ref.linear.weight = torch.nn.Parameter(clamped_weights)

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
                self.data: torch.Tensor = data
                self.__log_equation_context(
                    synaptic_weight_equation, dw_dt, spike, self.lif.mem())

    def train_inhibitory_from_layer(self, synaptic_update_type: SynapticUpdateType, spike: torch.Tensor,
                                    from_layer: Self) -> None:
        # expand the mask across the synaptic weight matrix
        mask = from_layer.inhibitory_mask_vec.unsqueeze(
            0).expand(self.layer_settings.size, -1)
        assert mask.shape == (
            self.layer_settings.size,
            from_layer.layer_settings.size)

        self.inhibitory_trace.apply(spike, self.layer_settings.dt)

        with torch.no_grad():
            x_i = self.inhibitory_trace.tracked_value().unsqueeze(
                2).expand(-1, -1, from_layer.layer_settings.size)
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

            dw_dt = self.layer_settings.learning_rate * \
                (first_term + second_term)
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

            weight_ref.linear.weight += dw_dt
            clamped_weights = torch.clamp(weight_ref.linear.weight, min=0)
            weight_ref.linear.weight = torch.nn.Parameter(clamped_weights)

    def train_synapses(self, spike: torch.Tensor, data: torch.Tensor) -> None:
        # recurrent connections always trained
        self.train_excitatory_from_layer(
            SynapticUpdateType.RECURRENT,
            spike,
            self.recurrent_filter_group,
            self,
            data)
        self.train_inhibitory_from_layer(
            SynapticUpdateType.RECURRENT, spike, self)

        if self.next_layer is not None:
            self.train_excitatory_from_layer(
                SynapticUpdateType.BACKWARD, spike, self.backward_filter_group, self.next_layer, data)
            self.train_inhibitory_from_layer(
                SynapticUpdateType.BACKWARD, spike, self.next_layer)

        # if prev layer is None then forward connections driven by data
        self.train_excitatory_from_layer(SynapticUpdateType.FORWARD, spike,
                                         self.forward_filter_group, self.prev_layer, data)

        # no forward connections from data are treated as inhibitory
        if self.prev_layer is not None:
            self.train_inhibitory_from_layer(
                SynapticUpdateType.FORWARD, spike, self.prev_layer)

        logging.debug(f"trained layer {self.layer_settings.layer_id} synapses")
