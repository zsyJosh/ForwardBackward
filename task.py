import math

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_scatter import scatter_add, scatter_min, scatter_mean, scatter_max

from torchdrug import core, layers, metrics, tasks
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.KnowledgeGraphCompletionCrop")
class KnowledgeGraphCompletionCrop(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, filtered_ranking=True,
                 fact_ratio=None, sample_weight=True, num_hop=2, num_neighbor=50, crop_then_negative=False,
                 random_edge_ratio=0, log_linear_neighbor=False):
        super(KnowledgeGraphCompletionCrop, self).__init__(
            model, criterion, metric, num_negative, margin, adversarial_temperature, strict_negative, filtered_ranking,
            fact_ratio, sample_weight)
        self.num_hop = num_hop
        self.num_neighbor = num_neighbor
        self.crop_then_negative = crop_then_negative
        self.random_edge_ratio = random_edge_ratio
        self.log_linear_neighbor = log_linear_neighbor

    @torch.no_grad()
    def get_subgraph(self, graph, h_index, t_index):
        # TODO: remove nodes that only belong to one entity's neighborhood
        # to reduce the size of the subgraph
        batch_size = len(h_index)
        nodes = torch.stack([h_index, t_index], dim=-1).flatten()
        num_nodes = torch.ones(batch_size, dtype=torch.long, device=self.device) * (nodes.numel() // batch_size)
        nodes, num_nodes = variadic_unique(nodes, num_nodes)
        node_visited = nodes
        num_node_visited = num_nodes
        edge_visited = torch.tensor([], dtype=torch.long, device=self.device)
        num_edge_visited = torch.zeros_like(num_node_visited)

        for i in range(self.num_hop):
            any = -torch.ones_like(nodes)
            node2sample = functional._size_to_index(num_nodes)

            pattern = torch.stack([nodes, any, any], dim=-1)
            edge_index, num_neighbors = graph.match(pattern)
            if isinstance(self.num_neighbor, int):
                max_neighbors = torch.ones_like(num_neighbors) * self.num_neighbor
            else:
                low, high = self.num_neighbor
                max_neighbors = torch.randint_like(num_neighbors, low, high + 1, device=self.device)
            edge_index, num_neighbors = variadic_downsample(edge_index, num_neighbors, max_neighbors)
            new_tails = graph.edge_list[edge_index, 1]
            num_new_tails = scatter_add(num_neighbors, node2sample, dim_size=len(num_nodes))
            edge_visited, num_edge_visited = \
                functional._extend(edge_visited, num_edge_visited, edge_index, num_new_tails)

            pattern = torch.stack([any, nodes, any], dim=-1)
            edge_index, num_neighbors = graph.match(pattern)
            if isinstance(self.num_neighbor, int):
                max_neighbors = torch.ones_like(num_neighbors) * self.num_neighbor
            else:
                # [low, high]
                low, high = self.num_neighbor
                if self.log_linear_neighbor:
                    rand = torch.rand(num_neighbors.shape, device=self.device)
                    max_neighbors = (rand * math.log((high + 1) / low)).exp() * low
                    max_neighbors = max_neighbors.long()
                else:
                    max_neighbors = torch.randint_like(num_neighbors, low, high + 1, device=self.device)
            edge_index, num_neighbors = variadic_downsample(edge_index, num_neighbors, max_neighbors)
            node2sample = functional._size_to_index(num_nodes)
            new_heads = graph.edge_list[edge_index, 0]
            num_new_heads = scatter_add(num_neighbors, node2sample, dim_size=len(num_nodes))
            edge_visited, num_edge_visited = \
                functional._extend(edge_visited, num_edge_visited, edge_index, num_new_heads)

            new_nodes, num_new_nodes = functional._extend(new_tails, num_new_tails, new_heads, num_new_heads)
            new_nodes, num_new_nodes = variadic_unique(new_nodes, num_new_nodes)
            new_nodes, num_new_nodes = variadic_except(new_nodes, num_new_nodes, node_visited, num_node_visited)
            node_visited, num_node_visited = functional._extend(node_visited, num_node_visited, new_nodes, num_new_nodes)

            nodes = new_nodes
            num_nodes = num_new_nodes

        if self.random_edge_ratio:
            random_edge = torch.rand(batch_size, graph.num_edge, device=self.device) < self.random_edge_ratio
            num_random_edge = random_edge.sum(dim=-1)
            random_edge = random_edge.nonzero()[:, 1]
            edge_visited, num_edge_visited = \
                functional._extend(edge_visited, num_edge_visited, random_edge, num_random_edge)
        edge_visited, num_edge_visited = variadic_unique(edge_visited, num_edge_visited)

        graph = multi_subgraph(graph, node_visited, num_node_visited, edge_visited, num_edge_visited)
        if h_index.ndim > 1:
            node_visited = node_visited.unsqueeze(-1)
        h_match = h_index[graph.node2graph] == node_visited
        h_index = scatter_max(h_match.long(), graph.node2graph, dim=0, dim_size=len(graph))[1]
        t_match = t_index[graph.node2graph] == node_visited
        t_index = scatter_max(t_match.long(), graph.node2graph, dim=0, dim_size=len(graph))[1]

        return graph, h_index, t_index

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)

        if all_loss is None:
            # test
            all_index = torch.arange(self.num_entity, device=self.device)
            t_preds = []
            h_preds = []
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.crop_then_negative:
                graph, pos_h_index, pos_t_index = self.get_subgraph(self.fact_graph, pos_h_index, pos_t_index)
            else:
                graph = self.fact_graph
            if self.strict_negative:
                neg_index = self._strict_negative(graph, pos_h_index, pos_t_index, pos_r_index)
            else:
                if len(graph) > 1:
                    neg_index = torch.rand(batch_size, self.num_negative, device=self.device) \
                                * graph.num_nodes.unsqueeze(-1) + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)
                else:
                    neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            if not self.crop_then_negative:
                graph, h_index, t_index = self.get_subgraph(self.fact_graph, h_index, t_index)

            if self.strict_negative:
                pattern = torch.stack([h_index[:, 1:], t_index[:, 1:], r_index[:, 1:]], dim=-1)
                pattern = pattern.flatten(0, -2)
                num_match = graph.match(pattern)[1]
                assert not num_match.any()
            if len(graph) > 1:
                range = torch.arange(batch_size, device=self.device)
                assert (graph.node2graph[h_index] == range.unsqueeze(-1)).all()
                assert (graph.node2graph[t_index] == range.unsqueeze(-1)).all()
            else:
                assert (h_index >= 0).all() and (h_index < graph.num_node).all()
                assert (t_index >= 0).all() and (t_index < graph.num_node).all()

            pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

            metric["subgraph #node"] = graph.num_nodes.float().mean()
            metric["subgraph #edge"] = graph.num_edges.float().mean()

        return pred

    @torch.no_grad()
    def _strict_negative(self, graph, pos_h_index, pos_t_index, pos_r_index):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        pattern = pattern[:batch_size // 2]
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        if len(graph) > 1:
            num_node = graph.num_nodes[:batch_size // 2].sum()
            t_mask = torch.ones(num_node, dtype=torch.bool, device=self.device)
            t_mask[t_truth_index] = 0
            neg_t_candidate = t_mask.nonzero()[:, 0]
            num_t_candidate = scatter_add(t_mask.long(), graph.node2graph[:num_node], dim_size=len(pattern))
            assert (num_t_candidate > 0).all()
        else:
            pos_index = functional._size_to_index(num_t_truth)
            t_mask = torch.ones(len(pattern), graph.num_node, dtype=torch.bool, device=self.device)
            t_mask[pos_index, t_truth_index] = 0
            neg_t_candidate = t_mask.nonzero()[:, 1]
            num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        pattern = pattern[batch_size // 2:]
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        if len(graph) > 1:
            num_node = graph.num_nodes[batch_size // 2:].sum()
            offset = graph.num_node - num_node
            h_mask = torch.ones(num_node, dtype=torch.bool, device=self.device)
            h_mask[h_truth_index - offset] = 0
            neg_h_candidate = h_mask.nonzero()[:, 0] + offset
            num_h_candidate = scatter_add(h_mask.long(), graph.node2graph[-num_node:] - batch_size // 2,
                                          dim_size=len(pattern))
            assert (num_h_candidate > 0).all()
        else:
            pos_index = functional._size_to_index(num_h_truth)
            h_mask = torch.ones(len(pattern), graph.num_node, dtype=torch.bool, device=self.device)
            h_mask[pos_index, h_truth_index] = 0
            neg_h_candidate = h_mask.nonzero()[:, 1]
            num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index


@R.register("tasks.KnowledgeGraphCompletionEx")
class KnowledgeGraphCompletionEx(tasks.KnowledgeGraphCompletion, core.Configurable):

    max_distance = 10

    def target(self, batch):
        mask, target = super(KnowledgeGraphCompletionEx, self).target(batch)

        graph = self.fact_graph
        node_in, node_out = graph.edge_list.t()[:2]
        node_in, node_out = torch.cat([node_in, node_out]), torch.cat([node_out, node_in])
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        range = torch.arange(batch_size, device=self.device)
        distance = torch.ones(graph.num_node, batch_size, dtype=torch.long, device=self.device) * self.max_distance
        distance[pos_h_index, range] = 0

        last = torch.zeros_like(distance)
        while not torch.equal(last, distance):
            last = distance
            distance = scatter_min(distance[node_in] + 1, node_out, dim=0, dim_size=graph.num_node)[0]
            distance = torch.min(distance, last)
        distance = distance[pos_t_index, range]

        return mask, target, distance.cpu()

    def evaluate(self, pred, target):
        mask, target, distance = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
        distance_type = distance.unique()

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                triplet_score = ranking.float().mean(dim=-1)
            elif _metric == "mrr":
                triplet_score = (1 / ranking.float()).mean(dim=-1)
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                triplet_score = (ranking <= threshold).float().mean(dim=-1)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)
            distance_score = scatter_mean(triplet_score, distance)
            score = triplet_score.mean()

            name = tasks._get_metric_name(_metric)
            for d in distance_type.tolist():
                metric["[distance=%d] %s" % (d, name)] = distance_score[d]
            metric[name] = score

        return metric


@R.register("tasks.InductiveRelationPrediction")
class InductiveRelationPrediction(tasks.Task, core.Configurable):

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="ce", metric="acc", num_mlp_layer=1):
        super(InductiveRelationPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.id2task = dataset.id2task
        self.task2id = dataset.task2id

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [dataset.num_relation])

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        for criterion, weight in self.criterion.items():
            if criterion == "ce":
                loss = F.cross_entropy(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        query = batch["query"]
        target = batch["target"]
        task = batch["task"]
        feature = self.model(graph, query, all_loss=all_loss, metric=metric)
        pred = self.mlp(feature)
        if all_loss is None:
            target = (target, task)
        return pred, target

    def evaluate(self, pred, target):
        target, task = target

        metric = {}
        for _metric in self.metric:
            if _metric == "acc":
                sample_score = (pred.argmax(dim=-1) == target).float()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            score = sample_score.mean()
            task_score = scatter_mean(sample_score, task, dim_size=len(self.id2task))
            name = tasks._get_metric_name(_metric)
            for i, task_type in enumerate(self.id2task):
                metric["[%s] %s" % (task_type, name)] = task_score[i]
            metric[name] = score

        return metric


def variadic_downsample(input, size, max_sample):
    perm = functional.variadic_randperm(size)
    perm = perm + (size.cumsum(0) - size).repeat_interleave(size)
    if isinstance(max_sample, int):
        new_size = size.clamp(max=max_sample)
    else:
        new_size = torch.min(size, max_sample)
    starts = size.cumsum(0) - size
    ends = starts + new_size
    mask = functional.multi_slice_mask(starts, ends, len(input))
    return input[perm[mask]], new_size


def variadic_sort(input, size, descending=False):
    index2sample = functional._size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))

    max = input.max().item()
    min = input.min().item()
    # special case: max = min
    offset = max - min + 1

    if descending:
        offset = -offset
    input_ext = input + offset * index2sample
    index = input_ext.argsort(dim=0, descending=descending)
    value = input.gather(0, index)
    index = index - (size.cumsum(0) - size)[index2sample]
    return value, index


def variadic_unique(input, size):
    assert input.dtype == torch.long
    index2sample = functional._size_to_index(size)

    max = input.max().item()
    min = input.min().item()
    # special case: max = min
    offset = max - min + 1

    input_ext = input + offset * index2sample
    input_ext = input_ext.unique()
    output = (input_ext - min) % offset + min
    index2sample = torch.div(input_ext - min, offset, rounding_mode="floor")
    new_size = index2sample.bincount(minlength=len(size))
    return output, new_size


def variadic_except(input1, size1, input2, size2):
    assert input1.dtype == torch.long and input2.dtype == torch.long
    assert size1.shape == size2.shape
    if len(input1) < len(input2):
        input1 = variadic_sort(input1, size1)[0]
    else:
        input2 = variadic_sort(input2, size2)[0]
    index2sample1 = functional._size_to_index(size1)
    index2sample2 = functional._size_to_index(size2)

    input = torch.cat([input1, input2])
    max = input.max().item()
    min = input.min().item()
    # special case: max = min
    offset = max - min + 1

    input1_ext = input1 + offset * index2sample1
    input2_ext = input2 + offset * index2sample2
    if len(input1) < len(input2):
        left = torch.bucketize(input2_ext, input1_ext)
        right = torch.bucketize(input2_ext, input1_ext, right=True)
        mask = ~functional.multi_slice_mask(left, right, len(input1))
    else:
        left = torch.bucketize(input1_ext, input2_ext)
        right = torch.bucketize(input1_ext, input2_ext, right=True)
        mask = right - left == 0
    output = input1[mask]
    size = functional.variadic_sum(mask.int(), size1).long()
    return output, size


def multi_subgraph(graph, node_index, num_nodes, edge_index=None, num_edges=None):
    batch_size = len(num_nodes)
    graph = graph.repeat(batch_size)
    node_index = node_index + (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_nodes)
    if edge_index is not None:
        edge_index = edge_index + (graph.num_cum_edges - graph.num_edges).repeat_interleave(num_edges)
        graph = graph.edge_mask(edge_index)

    return graph.subgraph(node_index)