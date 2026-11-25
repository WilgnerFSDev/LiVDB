import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNStack(nn.Module):
    def __init__(self, channels, num_relations, num_layers, aggr='add', dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([
            RGCNConv(channels, channels, num_relations=num_relations, aggr=aggr)
            for _ in range(num_layers)
        ])
        self.gru = nn.GRUCell(channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, edge_index, edge_type):
        h = x
        for conv in self.convs:
            m = conv(h, edge_index, edge_type)
            h = self.gru(m, h)
            if self.dropout is not None:
                h = self.dropout(h)
        return h


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=6):
        super().__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.num_timesteps = num_steps
        self.max_edge_types = max_edge_types

        self.input_proj = nn.Linear(input_dim, output_dim)

        self.ggnn = RGCNStack(
            channels=output_dim,
            num_relations=max_edge_types,
            num_layers=num_steps,
            aggr='add'
        )

        self.conv_l1 = nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv_l2 = nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(self.concat_dim, 1)
        self.mlp_y = nn.Linear(output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        _, features, edge_index, edge_type = batch.get_network_inputs(cuda=cuda)

        projected = self.input_proj(features)
        outputs   = self.ggnn(projected, edge_index, edge_type)

        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)

        Y_1 = self.maxpool1(F.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(F.relu(self.conv_l2(Y_1))).transpose(1, 2)

        Z_1 = self.maxpool1_for_concat(F.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(F.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg    = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result

    def forward_embedding(self, batch, cuda=False):
        _, features, edge_index, edge_type = batch.get_network_inputs(cuda=cuda)
        projected = self.input_proj(features)
        outputs   = self.ggnn(projected, edge_index, edge_type)
        h_i, _   = batch.de_batchify_graphs(outputs)
        Y_1 = self.maxpool1(F.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(F.relu(self.conv_l2(Y_1))).transpose(1, 2)
        return Y_2.mean(dim=1)


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super().__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.num_timesteps = num_steps
        self.max_edge_types = max_edge_types

        self.input_proj = nn.Linear(input_dim, output_dim)

        self.ggnn = RGCNStack(
            channels=output_dim,
            num_relations=max_edge_types,
            num_layers=num_steps,
            aggr='add',
            dropout=0.2
        )

        self.classifier = nn.Linear(output_dim, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        _, features, edge_index, edge_type = batch.get_network_inputs(cuda=cuda)

        projected = self.input_proj(features)
        outputs   = self.ggnn(projected, edge_index, edge_type)
        outputs   = torch.tanh(outputs)

        h_i, _   = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result   = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

    def forward_embedding(self, batch, cuda=False):
        _, features, edge_index, edge_type = batch.get_network_inputs(cuda=cuda)
        projected = self.input_proj(features)
        outputs   = self.ggnn(projected, edge_index, edge_type)
        outputs   = torch.tanh(outputs)
        h_i, _   = batch.de_batchify_graphs(outputs)
        return h_i.sum(dim=1)
