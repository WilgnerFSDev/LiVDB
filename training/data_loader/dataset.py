import copy
import json
import sys

import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug


class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target

        src_ids, edge_types, dst_ids = [], [], []
        for s, _type, t in edges:
            src_ids.append(s)
            edge_types.append(self.dataset.get_edge_type_number(_type))
            dst_ids.append(t)

        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        x = torch.FloatTensor(features)

        self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DataSet:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, test_src, train_src, valid_src):
        self.train_examples = self._read_file(train_src, "Train")
        if valid_src is not None:
            self.valid_examples = self._read_file(valid_src, "Validation")
        if test_src is not None:
            self.test_examples = self._read_file(test_src, "Test")

    def _read_file(self, src, split_name):
        debug(f'Reading {split_name} File!')
        examples = []
        if src.endswith('.jsonl'):
            with open(src) as fp:
                for line in tqdm(fp):
                    entry = json.loads(line)
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    if self.feature_size == 0:
                        self.feature_size = example.graph.x.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    examples.append(example)
        else:
            with open(src) as fp:
                data = json.load(fp)
                for entry in tqdm(data):
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    if self.feature_size == 0:
                        self.feature_size = example.graph.x.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    examples.append(example)
        return examples

    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(entry.graph)
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
