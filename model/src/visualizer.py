import networkx as nx
import matplotlib.pyplot as plt

from model.src.network import Net


class NetworkVisualizer:
    def __init__(self, net: Net, show_layer_views: bool = True) -> None:
        self.show_layer_views = show_layer_views

        self.net = net
        if show_layer_views:
            self.fig, self.axs = plt.subplots(
                1, len(self.net.layers) + 1, figsize=(12, 8))
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(12, 8))
            self.axs = [self.axs]
        self.summary_ax = self.axs[0]
        self.layer_axs = self.axs[1:]
        self.colors = {'excitatory': 'blue', 'inhibitory': 'orange',
                       'spiking_excitatory': 'green', 'spiking_inhibitory': 'red'}

    def update(self) -> None:
        self.summary_ax.clear()

        # Draw summary network
        G_summary = nx.DiGraph()
        for i, layer in enumerate(self.net.layers):
            for j in range(layer.layer_settings.size):
                neuron_color = self.colors['excitatory'] if layer.excitatory_mask_vec[j] else self.colors['inhibitory']
                if layer.lif.spike_moving_average.spike_rec[-1][0, j].item(
                ) > 0:
                    neuron_color = self.colors['spiking_excitatory'] if layer.excitatory_mask_vec[
                        j] else self.colors['spiking_inhibitory']
                G_summary.add_node((i, j), color=neuron_color)

        pos_summary = {}
        for node in G_summary.nodes:
            pos_summary[node] = (node[0], -node[1])
        node_colors_summary = [data['color']
                               for _, data in G_summary.nodes(data=True)]
        nx.draw_networkx_nodes(G_summary, pos_summary, node_color=node_colors_summary,
                               node_size=100, ax=self.summary_ax)
        self.summary_ax.set_title('Summary Network')
        self.summary_ax.set_xlim(-0.5, len(self.net.layers) - 0.5)
        self.summary_ax.set_ylim(
            -max(layer.layer_settings.size for layer in self.net.layers) + 0.5, 0.5)
        self.summary_ax.set_aspect('equal')
        self.summary_ax.set_axis_off()

        if self.show_layer_views:
            # Draw layer-specific views
            for i, layer in enumerate(self.net.layers):
                ax = self.layer_axs[i]
                ax.clear()

                G_layer = nx.DiGraph()

                # Add neurons
                for j in range(layer.layer_settings.size):
                    neuron_color = self.colors['excitatory'] \
                        if layer.excitatory_mask_vec[j] else self.colors['inhibitory']
                    if layer.lif.spike_moving_average.spike_rec[-1][0, j].item(
                    ) > 0:
                        neuron_color = self.colors['spiking_excitatory'] if layer.excitatory_mask_vec[
                            j] else self.colors['spiking_inhibitory']
                    G_layer.add_node((0, j), color=neuron_color)
                    G_layer.add_node((1, j), color=neuron_color)

                # Add recurrent connections
                for j in range(layer.layer_settings.size):
                    for k in range(layer.layer_settings.size):
                        weight = layer.recurrent_weights.weight()[j, k].item()
                        if weight != 0:
                            edge_color = self.colors['excitatory'] \
                                if layer.excitatory_mask_vec[k] else self.colors['inhibitory']
                            if layer.lif.spike_moving_average.spike_rec[-1][0, k].item(
                            ) > 0:
                                edge_color = self.colors['spiking_excitatory'] if layer.excitatory_mask_vec[
                                    k] else self.colors['spiking_inhibitory']
                            G_layer.add_edge(
                                (0, k), (1, j), color=edge_color, weight=abs(weight))

                # Add forward connections
                if i > 0:
                    prev_layer = self.net.layers[i - 1]
                    for j in range(layer.layer_settings.size):
                        for k in range(prev_layer.layer_settings.size):
                            weight = layer.forward_weights.weight()[
                                j, k].item()
                            if weight != 0:
                                edge_color = self.colors['excitatory'] \
                                    if prev_layer.excitatory_mask_vec[k] else self.colors['inhibitory']
                                if prev_layer.lif.spike_moving_average.spike_rec[-1][0, k].item(
                                ) > 0:
                                    edge_color = self.colors['spiking_excitatory'] if prev_layer.excitatory_mask_vec[
                                        k] else self.colors['spiking_inhibitory']
                                G_layer.add_edge(
                                    (-1, k), (0, j), color=edge_color, weight=abs(weight))
                                G_layer.add_node(
                                    (-1, k), color=edge_color)

                # Add backward connections
                if i < len(self.net.layers) - 1:
                    next_layer = self.net.layers[i + 1]
                    for j in range(layer.layer_settings.size):
                        for k in range(next_layer.layer_settings.size):
                            weight = layer.backward_weights.weight()[
                                j, k].item()
                            if weight != 0:
                                edge_color = self.colors['excitatory'] \
                                    if next_layer.excitatory_mask_vec[k] else self.colors['inhibitory']
                                if next_layer.lif.spike_moving_average.spike_rec[-1][0, k].item(
                                ) > 0:
                                    edge_color = self.colors['spiking_excitatory'] if next_layer.excitatory_mask_vec[
                                        k] else self.colors['spiking_inhibitory']
                                G_layer.add_edge(
                                    (2, k), (1, j), color=edge_color, weight=abs(weight))
                                G_layer.add_node(
                                    (2, k), color=edge_color)

                pos_layer = {}
                for node in G_layer.nodes:
                    pos_layer[node] = (node[0], -node[1])
                node_colors_layer = [data['color']
                                     for _, data in G_layer.nodes(data=True)]
                edge_colors_layer = [data['color']
                                     for _, _, data in G_layer.edges(data=True)]
                edge_weights_layer = [data['weight']
                                      for _, _, data in G_layer.edges(data=True)]

                nx.draw_networkx_nodes(
                    G_layer,
                    pos_layer,
                    node_color=node_colors_layer,
                    node_size=100,
                    ax=ax)
                nx.draw_networkx_edges(G_layer, pos_layer, edge_color=edge_colors_layer,
                                       width=edge_weights_layer, alpha=0.7, ax=ax)

                ax.set_title(f'Layer {i}')
                ax.set_xlim(-1.5, 2.5)
                ax.set_ylim(-layer.layer_settings.size + 0.5, 0.5)
                ax.set_aspect('equal')
                # ax.set_xticks([-1, 0, 1, 2],
                #               ['Prev Layer' if i > 0 else '', 'Layer', 'Layer',
                #                'Next Layer' if i < len(self.net.layers) - 1 else '']
                #               )
                # ax.set_axis_off()

        # Draw color key
        for i, (label, color) in enumerate(self.colors.items()):
            self.fig.text(
                0.9,
                0.9 - i * 0.05,
                f'{label}',
                color=color,
                ha='left',
                va='center')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)
