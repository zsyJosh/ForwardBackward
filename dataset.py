import os
import csv
import glob

from tqdm import tqdm

import torch
from torch.utils import data as torch_data
from torch_scatter import scatter_min

from torchdrug import core, data, datasets, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("datasets.Kinship")
class Kinship(data.KnowledgeGraphDataset):

    urls = [
        "https://raw.githubusercontent.com/Colinasda/KGdatasets/main/Kinship/train.txt",
        "https://raw.githubusercontent.com/Colinasda/KGdatasets/main/Kinship/valid.txt",
        "https://raw.githubusercontent.com/Colinasda/KGdatasets/main/Kinship/test.txt",
    ]

    md5s = [
        "e3f69f8d7c957ce1403f0d13d6bfedc7",
        "1a29862bd3b80e7ae758a022991ac477",
        "4130c202a819b5e42f8acde1f0fddf6e",
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "kinship_%s" % os.path.basename(url)
            txt_file = utils.download(url, self.path, save_file=save_file, md5=md5)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.FB15k237Distance")
class FB15k237Distance(datasets.FB15k237):

    max_distance = 10

    def __init__(self, path, train_range=None, valid_range=None, test_range=None, verbose=1):
        super(FB15k237Distance, self).__init__(path, verbose=verbose)

        train_range = train_range or (0, self.max_distance)
        valid_range = train_range or (0, self.max_distance)
        test_range = train_range or (0, self.max_distance)
        ranges = torch.tensor([train_range, valid_range, test_range])

        train_graph = self.graph.edge_mask(range(self.num_samples[0]))
        split = torch.repeat_interleave(torch.tensor(self.num_samples))
        node_in, node_out, relation = self.graph.edge_list.t()
        node_in = node_in.tolist()
        node_out = node_out.tolist()
        relation = relation.tolist()
        distance = []
        for i in tqdm(range(self.graph.num_edge)):
            pattern = torch.tensor([[node_in[i], node_out[i], -1], [node_out[i], node_in[i], -1]])
            edge_index = train_graph.match(pattern)[0]
            edge_mask = ~functional.as_mask(edge_index, train_graph.num_edge)
            graph = train_graph.edge_mask(edge_mask)
            d = self.pair_distance(graph, node_in[i], node_out[i])
            distance.append(d)
        distance = torch.tensor(distance)
        mask = (distance >= ranges[split, 0]) & (distance <= ranges[split, 1])
        self.samples = self.graph.edge_list[mask]
        self.num_samples = split[mask].bincount(minlength=3)

    def pair_distance(self, graph, source, target):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in, node_out = torch.cat([node_in, node_out]), torch.cat([node_out, node_in])
        node_in = node_in.cuda()
        node_out = node_out.cuda()

        distance = torch.ones(graph.num_node, dtype=torch.long, device="cuda") * self.max_distance
        distance[source] = 0
        last = torch.zeros_like(distance)
        while not torch.equal(last, distance) and distance[target] == self.max_distance:
            last = distance
            new_distance = scatter_min(distance[node_out], node_in, dim_size=graph.num_node)[0] + 1
            distance = torch.min(distance, new_distance)
        return distance[target].cpu()

    def single_source_distance(self, graph, source):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in, node_out = torch.cat([node_in, node_out]), torch.cat([node_out, node_in])
        node_in = node_in.cuda()
        node_out = node_out.cuda()

        distance = torch.ones(graph.num_node, dtype=torch.long, device="cuda") * self.max_distance
        distance[source] = 0
        last = torch.zeros_like(distance)
        while not torch.equal(last, distance):
            last = distance
            new_distance = scatter_min(distance[node_out], node_in, dim_size=graph.num_node)[0] + 1
            distance = torch.min(distance, new_distance)
        return distance.cpu()

    def all_pair_distance(self, graph):
        distances = []
        for i in tqdm(range(graph.num_node)):
            distance = self.single_source_distance(graph, i)
            distances.append(distance)
        distances = torch.stack(distances)
        return distances

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


@R.register("datasets.CLUTRR")
class CLUTRR(torch_data.Dataset, core.Configurable):

    url = "https://drive.google.com/u/1/uc?id=1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF"
    md5 = "9a79d1a6805ce30c0209decdfd4b63aa"
    task2file = {
        (3, None): "data_089907f8.zip",
        (4, None): "data_db9b8f04.zip",
        (3, "clean"): "data_7c5b0e70.zip",
        (3, "supporting"): "data_06b8f2a1.zip",
        (3, "irrelevant"): "data_523348e6.zip",
        (3, "disconnected"): "data_d83ecc3e.zip",
    }

    def __init__(self, path, length=3, noise=None, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, self.path, save_file="clutrr_data.zip", md5=self.md5)
        if noise:
            task_path = os.path.join(path, "clutrr_%d_%s" % (length, noise))
        else:
            task_path = os.path.join(path, "clutrr_%d" % length)
        file_name = self.task2file[(length, noise)]
        task_file = os.path.join(task_path, file_name)
        if not os.path.exists(task_path):
            os.makedirs(task_path)
            member = os.path.join("data_emnlp_final", file_name)
            file = utils.extract(zip_file, member=member)
            os.rename(file, task_file)
        utils.extract(task_file)

        train_files = glob.glob(os.path.join(task_path, "*_train.csv"))
        test_files = glob.glob(os.path.join(task_path, "*_test.csv"))

        self.load_csvs(train_files, test_files, verbose=verbose)

    def load_csvs(self, train_files, test_files, verbose=0):
        inv_relation_vocab = {}
        task2id = {}
        num_samples = []
        edge_lists = []
        queries = []
        targets = []
        tasks = []

        for csv_file in train_files + test_files:
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin)
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))

                num_sample = 0
                fields = next(reader)
                for values in reader:
                    for field, value in zip(fields, values):
                        if field == "story_edges":
                            edge_list = utils.literal_eval(value)
                            edge_list = torch.tensor(edge_list)
                        elif field == "edge_types":
                            relation = utils.literal_eval(value)
                            for r_token in relation:
                                if r_token not in inv_relation_vocab:
                                    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                            relation = [inv_relation_vocab[r] for r in relation]
                            relation = torch.tensor(relation)
                        elif field == "query_edge":
                            queries.append(utils.literal_eval(value))
                        elif field == "target":
                            r_token = value
                            if r_token not in inv_relation_vocab:
                                inv_relation_vocab[r_token] = len(inv_relation_vocab)
                            targets.append(inv_relation_vocab[r_token])
                        elif field == "task_name":
                            task = value[5:]
                            if task not in task2id:
                                task2id[task] = len(task2id)
                            tasks.append(task2id[task])

                    edge_list = torch.cat([edge_list, relation.unsqueeze(-1)], dim=-1)
                    edge_lists.append(edge_list)
                    num_sample += 1
            num_samples.append(num_sample)

        num_train = sum(num_samples[:len(train_files)])
        num_test = sum(num_samples[len(train_files):])
        self.num_samples = [num_train, num_test]
        self.edge_lists = edge_lists
        self.queries = queries
        self.targets = targets
        self.tasks = tasks

        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)
        id2task, task2id = self._standarize_vocab(None, task2id)
        self.relation_vocab = relation_vocab
        self.inv_relation_vocab = inv_relation_vocab
        self.id2task = id2task
        self.task2id = task2id

    def split(self):
        num_train, num_test = self.num_samples
        indices = torch.randperm(num_train).tolist()
        train_set = torch_data.Subset(self, indices[:int(num_train * 0.8)])
        valid_set = torch_data.Subset(self, indices[int(num_train * 0.8):])
        test_set = torch_data.Subset(self, range(num_train, num_train + num_test))
        return train_set, valid_set, test_set

    def _standarize_vocab(self, vocab, inverse_vocab):
        if vocab is not None:
            if isinstance(vocab, dict):
                assert set(vocab.keys()) == set(range(len(vocab))), "Vocab keys should be consecutive numbers"
                vocab = [vocab[k] for k in range(len(vocab))]
            if inverse_vocab is None:
                inverse_vocab = {v: i for i, v in enumerate(vocab)}
        if inverse_vocab is not None:
            assert set(inverse_vocab.values()) == set(range(len(inverse_vocab))), \
                "Inverse vocab values should be consecutive numbers"
            if vocab is None:
                vocab = sorted(inverse_vocab, key=lambda k: inverse_vocab[k])
        return vocab, inverse_vocab

    def __getitem__(self, index):
        edge_list = self.edge_lists[index]
        num_node = edge_list[:, :2].max() + 1
        return {
            "graph": data.Graph(edge_list, num_node=num_node, num_relation=self.num_relation),
            "query": torch.tensor(self.queries[index]),
            "target": self.targets[index],
            "task": self.tasks[index],
        }

    @property
    def num_relation(self):
        return len(self.relation_vocab)

    def __len__(self):
        return len(self.edge_lists)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#relation: %d" % self.num_relation,
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


@R.register("datasets.CoDEx")
class CoDEx(data.KnowledgeGraphDataset, core.Configurable):

    urls = [
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-%s/train.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-%s/valid.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-%s/test.txt",
    ]

    def __init__(self, path, size="m", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url in self.urls:
            url = url % size
            save_file = "codex-%s_%s" % (size, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
