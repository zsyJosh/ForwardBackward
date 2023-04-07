import torch
from torch import nn

from torchdrug import core, data, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import model


@R.register("model.BackwardForwardReasoning")
class BackwardForwardReasoning(nn.Module, core.Configurable):

    eps = 1e-8

    def __init__(self, backward_model, forward_model, num_relation, num_rule=1, num_test_rule=None, remove_one_hop=False,
                 num_recursion=1, num_test_recursion=None, num_mlp_layer=2, query_as_indicator=False,
                 update_frequency=1000, fusion_type="score"):
        super(BackwardForwardReasoning, self).__init__()
        self.backward_model = backward_model
        self.forward_model = forward_model
        self.num_relation = num_relation
        self.num_rule = num_rule
        self.num_test_rule = num_test_rule or num_rule
        self.remove_one_hop = remove_one_hop
        self.num_recursion = num_recursion
        self.num_test_recursion = num_test_recursion or num_recursion
        self.query_as_indicator = query_as_indicator
        self.update_frequency = update_frequency
        self.fusion_type = fusion_type
        self.counter = 0

        feature_dim = forward_model.output_dim + self.backward_model.input_dim
        self.mlp = layers.MultiLayerPerceptron(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def backward_reasoning(self, r_index, all_loss=None, metric=None):
        assert (r_index[:, [0]] == r_index).all()

        query = self.backward_model.embed(r_index[:, 0])
        input = query
        mask = torch.ones(query.shape[:-1], dtype=torch.bool, device=self.device)
        num_recursion = self.num_recursion if self.training else self.num_test_recursion
        for i in range(num_recursion):
            queries, masks = self.backward_model(input, all_loss, metric)
            input = queries
            mask = mask.unsqueeze(-1) & masks
        queries = queries.flatten(1, -2)
        masks = mask.flatten(1)
        return query, queries, masks

    def forward_reasoning(self, graph, h_index, t_index, query, queries, masks, all_loss=None, metric=None):
        assert (h_index[:, [0]] == h_index).all()

        if self.query_as_indicator:
            hidden = self.forward_model.embed(graph, h_index[:, 0], query)
        else:
            hidden = self.forward_model.embed(graph, h_index[:, 0])
        for query, mask in zip(queries.unbind(dim=-2), masks.unbind(dim=-1)):
            if isinstance(graph, data.PackedGraph):
                with graph.graph():
                    graph.query = query
                hidden = self.forward_model(graph, hidden, all_loss, metric)
            else:
                with graph.graph():
                    graph.query = query[mask]
                hidden[mask] = self.forward_model(graph, hidden[mask])
        feature = self.forward_model.readout(graph, hidden, t_index)
        return feature

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
            self.counter += 1
            if isinstance(self.backward_model, model.RuleFromModel) and self.counter % self.update_frequency == 0:
                self.backward_model.update_score(self.forward_model, self.mlp)

        if graph.num_relation != self.num_relation * 2:
            graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        num_rule = self.num_rule if self.training else self.num_test_rule
        if num_rule > 1:
            h_index = h_index.expand(num_rule, -1, -1).flatten(0, 1)
            t_index = t_index.expand(num_rule, -1, -1).flatten(0, 1)
            r_index = r_index.expand(num_rule, -1, -1).flatten(0, 1)
        query, queries, masks = self.backward_reasoning(r_index, all_loss, metric)
        feature = self.forward_reasoning(graph, h_index, t_index, query, queries, masks, all_loss, metric)

        query = query.unsqueeze(-2).expand_as(feature)
        feature = torch.cat([feature, query], dim=-1)
        if num_rule > 1 and self.fusion_type == "feature":
            feature = feature.unflatten(0, (num_rule, len(feature) // num_rule)).mean(dim=0)

        score = self.mlp(feature).squeeze(-1)
        if num_rule > 1 and self.fusion_type == "score":
            score = score.unflatten(0, (num_rule, len(score) // num_rule)).mean(dim=0)
        return score

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index
