import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class BatchGraph:
    def __init__(self):
        self.graphs = []
        self.num_of_subgraphs = 0
        self._batch_obj = None

    def add_subgraph(self, _g):
        self.graphs.append(_g)
        self.num_of_subgraphs += 1
        self._batch_obj = None

    @property
    def batch_obj(self):
        if self._batch_obj is None:
            self._batch_obj = Batch.from_data_list(self.graphs)
        return self._batch_obj

    def cuda(self, device=None):
        if self.batch_obj is not None:
            self._batch_obj = self.batch_obj.to(device if device else 'cuda')

    def de_batchify_graphs(self, features):
        dense_features, mask = to_dense_batch(features, self.batch_obj.batch)

        lengths = mask.sum(dim=1).to(device=features.device)

        return dense_features, lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')


class GGNNBatchGraph(BatchGraph):
    def get_network_inputs(self, cuda=False, device=None):
        if cuda:
            self.cuda(device=device)

        return self.batch_obj, self.batch_obj.x, self.batch_obj.edge_index, self.batch_obj.edge_attr
