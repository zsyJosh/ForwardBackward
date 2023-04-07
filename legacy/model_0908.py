from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean, segment_add_coo

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import layer


@R.register("model.LogicNN")
class LogicMessagePassingNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, max_triangle=None, message_func="distmult",
                 aggregate_func="sum", short_cut=False, layer_norm=False, activation="relu", concat_hidden=False,
                 num_mlp_layer=2, remove_one_hop=False, independent_fact=False, dependent=False, only_init_loop=False,
                 concat_query=False, no_concat_input=False, dependent_fact_init=False, edge_dropout=0,
                 triangle_dropout=0, relation_as_fact=False, pre_activation=False, separate_fact_query=False,
                 undirected_ac=False, reconstruction_weight=0, min_coverage=0, min_confidence=0,
                 dependent_add=False, dependent_cat=False, query_fuse=False, layer_pinpeline=False):
        super(LogicMessagePassingNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.max_triangle = max_triangle or {}
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.independent_fact = independent_fact
        self.only_init_loop = only_init_loop
        self.concat_query = concat_query
        self.no_concat_input = no_concat_input
        self.dependent_fact_init = dependent_fact_init
        self.edge_dropout = edge_dropout
        self.relation_as_fact = relation_as_fact
        self.undirected_ac = undirected_ac
        self.reconstruction_weight = reconstruction_weight
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.layer_pinpeline = layer_pinpeline

        self.layers = nn.ModuleList()
        if self.layer_pinpeline:
            self.query_layers = nn.ModuleList()
            dependent = True
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.LogicMessagePassingConv(self.dims[i], self.dims[i + 1], num_relation * 2,
                                                             message_func, aggregate_func, layer_norm, activation,
                                                             dependent, triangle_dropout, pre_activation,
                                                             separate_fact_query, dependent_add, dependent_cat, query_fuse))
            if self.layer_pinpeline:
                self.query_layers.append(layer.LogicMessagePassingConv(self.dims[i], self.dims[i + 1], num_relation * 2,
                                                                 message_func, aggregate_func, layer_norm, activation,
                                                                 False, triangle_dropout, pre_activation,
                                                                 separate_fact_query, dependent_add, dependent_cat,
                                                                 query_fuse))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim * (1 - no_concat_input)
        self.query = nn.Embedding(num_relation * 2, input_dim)
        if independent_fact:
            self.layer_facts = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_facts.append(nn.Embedding(num_relation * 2, self.dims[i]))
        if dependent_fact_init:
            self.fact_linear = nn.Linear(input_dim, num_relation * 2 * input_dim)
        else:
            self.fact = nn.Embedding(num_relation * 2, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        # mimic NBFNet's remove easy edges
        # batch_size = len(h_index)
        # h_index = h_index.flatten().expand(batch_size, -1)
        # t_index = t_index.flatten().expand(batch_size, -1)
        # r_index = r_index.flatten().expand(batch_size, -1)

        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset.unsqueeze(-1)
        t_index = t_index + offset.unsqueeze(-1)
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        if self.remove_one_hop:
            any = -torch.ones_like(h_index_ext)
            pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            if graph.num_relation == self.num_relation:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            elif graph.num_relation == self.num_relation * 2:
                r_index_ext = torch.cat([r_index, r_index + self.num_relation], dim=-1)
                pattern = torch.stack([h_index_ext, t_index_ext, r_index_ext], dim=-1)
            else:
                raise ValueError
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        if self.edge_dropout > 0:
            is_dropped = torch.rand(graph.num_edge, device=self.device) <= self.edge_dropout
            edge_mask = edge_mask & ~is_dropped
        graph = graph.edge_mask(edge_mask)

        # incremental triangle computation
        if hasattr(graph, "edge_ab"):
            mask = (graph.edge_ab >= 0) & (graph.edge_bc >= 0) & (graph.edge_ac >= 0)
            with graph.edge_reference():
                graph.edge_ab = graph.edge_ab[mask]
                graph.edge_bc = graph.edge_bc[mask]
                graph.edge_ac = graph.edge_ac[mask]

        return graph

    def add_query_specific(self, graph, h_index, t_index, r_index):
        all_index = torch.arange(graph.num_nodes[0], device=self.device)
        h_index, t_index = torch.meshgrid(h_index[:, 0], all_index)
        r_index = r_index[:, [0]].expand_as(h_index)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset.unsqueeze(-1)
        t_index = t_index + offset.unsqueeze(-1)
        assert (h_index < graph.num_cum_nodes.unsqueeze(-1)).all()
        assert (h_index >= (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
        assert (t_index < graph.num_cum_nodes.unsqueeze(-1)).all()
        assert (t_index >= (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
        edge_list = torch.stack([h_index, t_index, r_index], dim=-1)
        edge_list = edge_list.flatten(0, -2)
        edge_weight = torch.ones(len(edge_list), device=self.device)
        not_query = torch.zeros(graph.num_edge, dtype=torch.bool, device=self.device)
        is_query = torch.ones(len(edge_list), dtype=torch.bool, device=self.device)
        num_query_specific = torch.ones(len(h_index), dtype=torch.long, device=self.device) * graph.num_nodes[0]

        edge_list, num_edges = functional._extend(graph.edge_list, graph.num_edges, edge_list, num_query_specific)
        edge_weight = functional._extend(graph.edge_weight, graph.num_edges, edge_weight, num_query_specific)[0]
        is_query = functional._extend(not_query, graph.num_edges, is_query, num_query_specific)[0]
        start = num_edges.cumsum(0) - num_edges
        end = start + graph.num_edges
        mask = functional.multi_slice_mask(start, end, num_edges.sum())
        data_dict, meta_dict = graph.data_by_meta(exclude="edge")
        offsets = graph._get_offsets(graph.num_nodes, num_edges)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                            num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)

        # incremental triangle computation
        if hasattr(graph, "edge_ab"):
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            # fact + fact -> query
            max_ffq = self.max_triangle.get("ffq", None)
            ffq_ab, ffq_bc, ffq_ac = single_source_triangle_list(
                graph, edge_ab=~is_query, edge_bc=~is_query, edge_ac=is_query, max_triangle_per_edge=max_ffq)
            # query + fact -> query
            max_qfq = self.max_triangle.get("qfq", None)
            qfq_ab, qfq_bc, qfq_ac = single_source_triangle_list(
                graph, edge_ab=is_query, edge_ac=is_query, max_triangle_per_edge=max_qfq)
            if self.layer_pinpeline:
                graph_fff = graph.clone()
                fff_edge_ab = mapping[graph.edge_ab]
                fff_edge_bc = mapping[graph.edge_bc]
                fff_edge_ac = mapping[graph.edge_ac]
                with graph_fff.edge_reference():
                    graph_fff.edge_ab = fff_edge_ab
                    graph_fff.edge_bc = fff_edge_bc
                    graph_fff.edge_ac = fff_edge_ac
                graph_query = graph.clone()
                query_edge_ab = torch.cat([ffq_ab, qfq_ab])
                query_edge_bc = torch.cat([ffq_bc, qfq_bc])
                query_edge_ac = torch.cat([ffq_ac, qfq_ac])
                with graph_query.edge_reference():
                    graph_query.edge_ab = query_edge_ab
                    graph_query.edge_bc = query_edge_bc
                    graph_query.edge_ac = query_edge_ac
            else:
                edge_ab = torch.cat([mapping[graph.edge_ab], ffq_ab, qfq_ab])
                edge_bc = torch.cat([mapping[graph.edge_bc], ffq_bc, qfq_bc])
                edge_ac = torch.cat([mapping[graph.edge_ac], ffq_ac, qfq_ac])
                with graph.edge_reference():
                    graph.edge_ab = edge_ab
                    graph.edge_bc = edge_bc
                    graph.edge_ac = edge_ac

        if self.layer_pinpeline:
            with graph_fff.edge():
                graph_fff.is_query = is_query
            with graph_fff.graph():
                graph_fff.query = r_index[:, 0]
            with graph_query.edge():
                graph_query.is_query = is_query
            with graph_query.graph():
                graph_query.query = r_index[:, 0]
            return graph_fff, graph_query
        else:
            with graph.edge():
                graph.is_query = is_query
            with graph.graph():
                graph.query = r_index[:, 0]

            return graph

    def relation_as_fact_graph(self, graph):
        batch_size = len(graph)
        relation = torch.arange(graph.num_relation, device=self.device).repeat(batch_size)
        entity = torch.zeros_like(relation)
        edge_list = torch.stack([entity, entity, relation], dim=-1)
        edge_weight = torch.ones(len(relation), device=self.device)
        num_edges = torch.ones(batch_size, dtype=torch.long, device=self.device) * graph.num_relation
        query_list = graph.edge_list[graph.is_query]
        query_weight = graph.edge_weight[graph.is_query]
        query2graph = graph.edge2graph[graph.is_query]
        num_queries = segment_add_coo(torch.ones_like(query2graph), query2graph, dim_size=batch_size)
        edge_list = functional._extend(edge_list, num_edges, query_list, num_queries)[0]
        edge_weight, num_edges = functional._extend(edge_weight, num_edges, query_weight, num_queries)
        num_cum_edges = num_edges.cumsum(0)
        is_query = functional.multi_slice_mask(num_cum_edges - num_queries, num_cum_edges, num_edges.sum())

        relation = graph.edge_list[:, 2]
        mapping = torch.arange(graph.num_edge, device=self.device)
        mapping[graph.is_query] = mapping[graph.is_query] + \
                                  (num_cum_edges - graph.num_cum_edges).repeat_interleave(num_queries)
        mapping[~graph.is_query] = relation[~graph.is_query] + \
                                   (num_cum_edges - num_edges).repeat_interleave(graph.num_edges - num_queries)
        data_dict, meta_dict = graph.data_by_meta(exclude="edge")
        offsets = graph._get_offsets(graph.num_nodes, num_edges)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                            num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)
        with graph.edge():
            graph.is_query = is_query
        with graph.edge_reference():
            graph.edge_ab = mapping[graph.edge_ab]
            graph.edge_bc = mapping[graph.edge_bc]
            graph.edge_ac = mapping[graph.edge_ac]

        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def get_input_embedding(self, graph):
        relation = graph.edge_list[:, 2]
        if self.dependent_fact_init:
            query = self.query(graph.query)
            fact = self.fact_linear(query).view(-1, self.num_relation * 2, self.dims[0])
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
        else:
            fact = self.fact(relation)
        query = self.query(relation)
        input = torch.where(graph.is_query.unsqueeze(-1), query, fact)
        if self.only_init_loop:
            h_index, t_index = graph.edge_list.t()[:2]
            mask = graph.is_query & (h_index != t_index)
            input = torch.where(mask.unsqueeze(-1), torch.zeros_like(input), input)

        if hasattr(graph, "query"):
            with graph.graph():
                graph.query = self.query(graph.query)
        return input

    def forward_chaining(self, graph, input, all_loss=None, metric=None, graph_fff=None, graph_query=None):
        # check_triangles(graph, graph.edge_ab, graph.edge_bc, graph.edge_ac)
        with graph.edge():
            graph.boundary = input

        if metric is not None:
            triangle_type = (graph.is_query[graph.edge_ab] * 2 + graph.is_query[graph.edge_bc]) * 2 \
                            + graph.is_query[graph.edge_ac]
            num_triangle = triangle_type.bincount(minlength=8)
            metric["#fact + fact -> fact"] = num_triangle[0b000] / len(graph)
            metric["#fact + fact -> query"] = num_triangle[0b001] / len(graph)
            metric["#query + fact -> query"] = num_triangle[0b101] / len(graph)
            metric["#query + query -> query"] = num_triangle[0b111] / len(graph)
            metric["#triangle"] = num_triangle.sum() / len(graph)
            metric["#variable"] = graph.num_edge / len(graph)

        hiddens = []
        layer_input = input

        for i, layer in enumerate(self.layers):
            if self.independent_fact:
                relation = graph.edge_list[:, 2]
                layer_fact = self.layer_facts[i](relation)
                layer_input = torch.where(graph.is_query.unsqueeze(-1), layer_input, layer_fact)
            if self.layer_pinpeline:
                hidden = self.query_layers[i](graph_fff, layer_input)
                hidden = layer(graph_query, hidden)
            else:
                hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.reconstruction_weight > 0 and all_loss is not None:
            assert self.concat_query and not self.no_concat_input
            relation = graph.edge_list[:, 2]
            target = torch.randint_like(relation, 0, 2).float()
            # corrupt the relation
            offset = torch.randint_like(relation, 1, graph.num_relation)
            corrupted = (relation + offset) % graph.num_relation
            relation = torch.where(target > 0.5, relation, corrupted)
            query = self.query(relation)
            if self.concat_hidden:
                output = torch.cat(hiddens + [query], dim=-1)
            else:
                output = torch.cat([hiddens[-1], query], dim=-1)
            pred = self.mlp(output).squeeze(-1)

            is_fact = ~graph.is_query
            pred = pred[is_fact]
            target = target[is_fact]
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            loss = scatter_mean(loss, graph.edge2graph[is_fact], dim_size=len(graph))
            loss = loss.mean()
            metric["fact reconstruction loss"] = loss
            all_loss += loss * self.reconstruction_weight

        if self.concat_query:
            input = graph.query.repeat_interleave(graph.num_edges, dim=0)
        if self.concat_hidden:
            if self.no_concat_input:
                output = torch.cat(hiddens, dim=-1)
            else:
                output = torch.cat(hiddens + [input], dim=-1)
        else:
            if self.no_concat_input:
                output = hiddens[-1]
            else:
                output = torch.cat([hiddens[-1], input], dim=-1)

        return output

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        assert graph.num_relation == self.num_relation

        graph = graph.undirected(add_inverse=True)
        for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
                print(">>>>>>>>>>>>>>>>>>>>>>>")
                print("coverage >= %g, confidence >= %g" % (i, t))
                rel_ab, rel_bc, rel_ac, coverage, confidence = rule_mining(graph, min_coverage=i, min_confidence=t)
                print("#rule: %d" % len(rel_ab))
                a_index, c_index, rel_ac = rule_inference(graph, rel_ab, rel_bc, rel_ac)
                print("#prediction: %d" % len(a_index))
                pattern = torch.stack([a_index, c_index, rel_ac], dim=-1)
                num_match = graph.match(pattern)[1]
                print("#new: %d" % (num_match == 0).sum())

        if not hasattr(self, "fff_triangles") or (self.graph.edge_list != graph.edge_list).any():
            self.graph = graph
            graph = graph.undirected(add_inverse=True)
            # fact + fact -> fact
            max_fff = self.max_triangle.get("fff", None)
            self.fff_triangles = triangle_list(graph, max_triangle_per_edge=max_fff, undirected_ac=self.undirected_ac)
            edge_ab, edge_bc, edge_ac = self.fff_triangles

            if self.min_coverage > 0 or self.min_confidence > 0:
                # filter triangles with mined logic rules
                rel_ab, rel_bc, rel_ac, coverage, confidence = rule_mining(graph, self.min_coverage, self.min_confidence)
                rule = tuple2scalar((rel_ab, rel_bc, rel_ac),
                                    (graph.num_relation, graph.num_relation, graph.num_relation))
                rule = rule.sort()[0]
                rel_ab = graph.edge_list[edge_ab, 2]
                rel_bc = graph.edge_list[edge_bc, 2]
                rel_ac = graph.edge_list[edge_ac, 2]
                grounding = tuple2scalar((rel_ab, rel_bc, rel_ac),
                                         (graph.num_relation, graph.num_relation, graph.num_relation))
                left = torch.bucketize(grounding, rule)
                right = torch.bucketize(grounding, rule, right=True)
                mask = right > left
                edge_ab = edge_ab[mask]
                edge_bc = edge_bc[mask]
                edge_ac = edge_ac[mask]
                self.fff_triangles = (edge_ab, edge_bc, edge_ac)
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
                '''
                graph.edge_ab = torch.tensor([], dtype=torch.long)
                graph.edge_bc = torch.tensor([], dtype=torch.long)
                graph.edge_ac = torch.tensor([], dtype=torch.long)
                '''
            self.undirected_graph = graph
        graph = self.undirected_graph

        batch_size = len(h_index)
        # cache the batch graph, so that we don't need to compute perfect hashing every time
        if not hasattr(self, "batch_size") or batch_size != self.batch_size:
            self.batch_size = batch_size
            self.batch_graph = graph.repeat(batch_size)
        graph = self.batch_graph
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (graph.num_nodes[0] == graph.num_nodes).all()
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        if self.layer_pinpeline:
            graph_fff, graph_query = self.add_query_specific(graph, h_index, t_index, r_index)
        else:
            graph = self.add_query_specific(graph, h_index, t_index, r_index)
        if self.relation_as_fact:
            if self.layer_pinpeline:
                new_graph_fff = self.relation_as_fact_graph(graph_fff)
                new_graph_query = self.relation_as_fact_graph(graph_query)
                assert (new_graph_fff.edge_list[new_graph_fff.edge_ab, 2] == graph_fff.edge_list[graph_fff.edge_ab, 2]).all()
                assert (new_graph_fff.edge_list[new_graph_fff.edge_bc, 2] == graph_fff.edge_list[graph_fff.edge_bc, 2]).all()
                assert (new_graph_fff.edge_list[new_graph_fff.edge_ac, 2] == graph_fff.edge_list[graph_fff.edge_ac, 2]).all()
                assert (new_graph_query.edge_list[new_graph_query.edge_ab, 2] == graph_query.edge_list[graph_query.edge_ab, 2]).all()
                assert (new_graph_query.edge_list[new_graph_query.edge_bc, 2] == graph_query.edge_list[graph_query.edge_bc, 2]).all()
                assert (new_graph_query.edge_list[new_graph_query.edge_ac, 2] == graph_query.edge_list[graph_query.edge_ac, 2]).all()
                graph_fff = new_graph_fff
                graph_query = new_graph_query
            else:
                new_graph = self.relation_as_fact_graph(graph)
                assert (new_graph.edge_list[new_graph.edge_ab, 2] == graph.edge_list[graph.edge_ab, 2]).all()
                assert (new_graph.edge_list[new_graph.edge_bc, 2] == graph.edge_list[graph.edge_bc, 2]).all()
                assert (new_graph.edge_list[new_graph.edge_ac, 2] == graph.edge_list[graph.edge_ac, 2]).all()
                graph = new_graph
        if self.layer_pinpeline:
            input = self.get_input_embedding(graph_query)
            output = self.forward_chaining(graph_query, input, all_loss, metric, graph_fff, graph_query)
            index = t_index + (graph_query.num_cum_edges - graph_query.num_nodes).unsqueeze(-1)
            assert graph_query.is_query[index].all()
            assert (graph_query.edge_list[index, 1] == t_index + (graph_query.num_cum_nodes - graph_query.num_nodes).unsqueeze(-1)).all()
        else:
            input = self.get_input_embedding(graph)
            output = self.forward_chaining(graph, input, all_loss, metric)
            index = t_index + (graph.num_cum_edges - graph.num_nodes).unsqueeze(-1)
            assert graph.is_query[index].all()
            assert (graph.edge_list[index, 1] == t_index + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
        feature = output[index]

        score = self.mlp(feature).squeeze(-1)
        return score.view(h_index.shape)


def variadic_sum(input, size):
    index2sample = functional._size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_add(input, index2sample, dim=0, dim_size=len(size))
    return value


def variadic_repeat_interleave(input, size, repeats):
    new_size = size.repeat_interleave(repeats)
    num_element = new_size.sum()
    batch_size = repeats.sum()
    new_cum_size = new_size.cumsum(0)

    # special case 1: size[i] may be 0
    # special case 2: repeats[i] may be 0
    cum_repeats_shifted = repeats.cumsum(0) - repeats
    sample_mask = cum_repeats_shifted < batch_size
    cum_repeats_shifted = cum_repeats_shifted[sample_mask]

    index = new_cum_size - new_size
    index = torch.cat([index, index[cum_repeats_shifted]])
    value = torch.cat([-new_size, size[sample_mask]])
    mask = index < num_element
    element_index = scatter_add(value[mask], index[mask], dim_size=num_element)
    element_index = (element_index + 1).cumsum(0) - 1

    return input[element_index], new_size


def variadic_index(input, size, index):
    cum_size = size.cumsum(0)

    new_size = size[index]
    new_cum_size = new_size.cumsum(0)
    range = torch.arange(new_size.sum(), device=size.device)
    element_index = range + (cum_size[index] - new_cum_size).repeat_interleave(new_size)

    return input[element_index], new_size


def variadic_unique(input, size):
    assert input.dtype == torch.long
    index2sample = functional._size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))

    max = input.max().item()
    min = input.min().item()
    offset = max - min + 1
    input_ext = input + offset * index2sample
    input_ext = input_ext.unique()
    index2sample = (input_ext - min) // offset
    input = (input_ext - min) % offset + min
    size = segment_add_coo(index2sample, dim_size=len(size))
    return input, size


def check_triangles(graph, edge_ab, edge_bc, edge_ac):
    node_a1, node_b1 = graph.edge_list[edge_ab].t()[:2]
    node_b2, node_c2 = graph.edge_list[edge_bc].t()[:2]
    node_a2, node_c1 = graph.edge_list[edge_ac].t()[:2]
    assert (node_a1 == node_a2).all()
    assert (node_b1 == node_b2).all()
    assert (node_c1 == node_c2).all()


def count_triangles(graph):
    # brute force
    adjacency = graph.adjacency.to_dense().long().sum(dim=-1)
    # matrix multiplication not implemented for long on GPUs
    adjacency = adjacency.cpu()
    num_path = adjacency.t() @ adjacency @ adjacency
    num_triangle = torch.diag(num_path).sum()
    return num_triangle


def tuple2scalar(tensors, ranges):
    assert len(tensors) == len(ranges)

    scalar = torch.zeros_like(tensors[0])
    for t, r in zip(tensors, ranges):
        scalar = scalar * r + t
    return scalar

def scalar2tuple(scalar, ranges):
    product = 1
    for r in ranges:
        product *= r
    assert scalar.numel() == 0 or scalar.max() < product

    tensors = []
    for r in reversed(ranges):
        tensors.append(scalar % r)
        scalar = scalar // r
    return tensors[::-1]


def triangle_list(graph, edge_ab=None, edge_bc=None, edge_ac=None, max_triangle_per_edge=None, undirected_ac=False):
    assert graph.num_node.item() ** 2 <= torch.iinfo(torch.long).max
    assert graph.num_node.item() * graph.num_edge.item() <= torch.iinfo(torch.long).max

    if edge_ab is None:
        edge_ab = torch.arange(graph.num_edge, device=graph.device)
    elif edge_ab.dtype == torch.bool:
        edge_ab = edge_ab.nonzero().squeeze(-1)
    if edge_bc is None:
        edge_bc = torch.arange(graph.num_edge, device=graph.device)
    elif edge_bc.dtype == torch.bool:
        edge_bc = edge_bc.nonzero().squeeze(-1)
    if edge_ac is None:
        edge_ac = torch.arange(graph.num_edge, device=graph.device)
    elif edge_ac.dtype == torch.bool:
        edge_ac = edge_ac.nonzero().squeeze(-1)

    # find all triangles that satisfy a -> c and a -> b -> c
    a_index, b_index = graph.edge_list[edge_ab, :2].t()
    ab_degree_a = a_index.bincount(minlength=graph.num_node)
    ab_order = (a_index * graph.num_node + b_index).argsort()
    a_neighbor_b = b_index[ab_order]

    b_index, c_index = graph.edge_list[edge_bc, :2].t()
    bc_degree_c = c_index.bincount(minlength=graph.num_node)
    bc_order = (c_index * graph.num_node + b_index).argsort()
    c_neighbor_b = b_index[bc_order]

    # ac_c_neighbor_b: b that satisfies a -> c and a -> b
    # ac_a_neighbor_b: b that satisfies a -> c and b -> c
    a_index, c_index = graph.edge_list[edge_ac, :2].t()
    if undirected_ac:
        edge_ac = edge_ac.repeat(2)
        a_index_ext = torch.cat([a_index, c_index])
        c_index_ext = torch.cat([c_index, a_index])
        a_index, c_index = a_index_ext, c_index_ext
    ac_a_neighbor_b, ac_degree_a = variadic_index(a_neighbor_b, ab_degree_a, a_index)
    ac_c_neighbor_b, ac_degree_c = variadic_index(c_neighbor_b, bc_degree_c, c_index)
    ac_ab_order = variadic_index(ab_order, ab_degree_a, a_index)[0]
    ac_bc_order = variadic_index(bc_order, bc_degree_c, c_index)[0]
    assert (ac_c_neighbor_b < graph.num_node).all()
    assert (ac_a_neighbor_b < graph.num_node).all()
    boundary = ac_c_neighbor_b + graph.num_node * functional._size_to_index(ac_degree_c)
    assert (boundary.diff() >= 0).all()
    key = ac_a_neighbor_b + graph.num_node * functional._size_to_index(ac_degree_a)
    left = torch.bucketize(key, boundary)
    right = torch.bucketize(key, boundary, right=True)
    count = right - left
    count_per_ac = variadic_sum(count, ac_degree_a)
    sample_per_ac = count_per_ac
    if max_triangle_per_edge is not None:
        sample_per_ac = sample_per_ac.clamp(max=max_triangle_per_edge)
        rand = torch.rand(sample_per_ac.sum(), device=graph.device)
        randint = (rand * count_per_ac.repeat_interleave(sample_per_ac)).long()
    else:
        randint = functional.variadic_arange(sample_per_ac)

    offset = count_per_ac.max()
    cum_count_shifted = (count.cumsum(0) - count) - \
                        (count_per_ac.cumsum(0) - count_per_ac).repeat_interleave(ac_degree_a)
    assert (cum_count_shifted <= offset).all()
    assert (randint < offset).all()
    boundary = cum_count_shifted + offset * functional._size_to_index(ac_degree_a)
    key = randint + offset * functional._size_to_index(sample_per_ac)
    index = torch.bucketize(key, boundary, right=True) - 1

    edge_ab = edge_ab[ac_ab_order[index]]
    # assert (randint >= cum_count_shifted[index]).all()
    # assert (randint < cum_count_shifted[index] + count[index]).all()
    edge_bc = edge_bc[ac_bc_order[key - boundary[index] + left[index]]]
    edge_ac = edge_ac.repeat_interleave(sample_per_ac)

    # sanity check
    # check_triangles(graph, edge_ab, edge_bc, edge_ac)

    return edge_ab, edge_bc, edge_ac


def rule_mining(graph, min_coverage=0.1, min_confidence=0.1):
    # AMIE algorithm
    assert graph.num_relation ** 2 * graph.num_edge <= torch.iinfo(torch.long).max
    assert (graph.num_node * graph.num_relation) ** 2 <= torch.iinfo(torch.long).max
    assert graph.num_node * graph.num_relation ** 3 <= torch.iinfo(torch.long).max
    edge_ab, edge_bc, edge_ac = triangle_list(graph)
    # print("per triangle variable: %g GiB" % (edge_ac.numel() * edge_ac.element_size() / 1024 ** 3))
    rel_ab = graph.edge_list[edge_ab, 2]
    rel_bc = graph.edge_list[edge_bc, 2]
    # deduplicate the body groundings w.r.t. edge_ac
    # each edge_ac can at most have one grounding for a given rule body
    grounding = tuple2scalar((rel_ab, rel_bc, edge_ac), (graph.num_relation, graph.num_relation, graph.num_edge))
    grounding = grounding.unique()
    rel_ab, rel_bc, edge_ac = scalar2tuple(grounding, (graph.num_relation, graph.num_relation, graph.num_edge))
    rel_ac = graph.edge_list[edge_ac, 2]
    body = tuple2scalar((rel_ab, rel_bc), (graph.num_relation, graph.num_relation))
    body_set = body.unique()
    rule = tuple2scalar((rel_ab, rel_bc, rel_ac), (graph.num_relation, graph.num_relation, graph.num_relation))
    rule_set = rule.unique()
    # discretize body_set & rule_set
    # body2id = data.Dictionary(body_set, torch.arange(len(body_set), device=graph.device))  # slow
    # rule2id = data.Dictionary(rule_set, torch.arange(len(rule_set), device=graph.device))  # slow
    body2id = -torch.ones(graph.num_relation ** 2, dtype=torch.long, device=graph.device)  # fast, but cost memory
    body2id[body_set] = torch.arange(len(body_set), device=graph.device)
    rule2id = -torch.ones(graph.num_relation ** 3, dtype=torch.long, device=graph.device)  # fast, but cost memory
    rule2id[rule_set] = torch.arange(len(rule_set), device=graph.device)
    # num_support = rule2id[rule].bincount(minlength=len(rule_set))  # slow
    ones = torch.ones(len(rule), dtype=torch.int, device=graph.device)
    num_support = scatter_add(ones, rule2id[rule], dim_size=len(rule_set)).long()  # fast
    # num_head = graph.edge_list[:, 2].bincount(minlength=graph.num_relation)  # slow
    ones = torch.ones(graph.num_edge, dtype=torch.int, device=graph.device)
    num_head = scatter_add(ones, graph.edge_list[:, 2], dim_size=graph.num_relation).long()  # fast
    rel_ac = scalar2tuple(rule_set, (graph.num_relation, graph.num_relation, graph.num_relation))[2]
    coverage = num_support / num_head[rel_ac]
    del edge_ab, edge_bc, edge_ac, rel_ab, rel_bc, rel_ac, body, rule, grounding, body_set, ones, num_head

    # enumerate all 2-hop random walks a -> b -> c
    b_index, c_index, rel_bc = graph.edge_list.t()
    bc_degree_b = b_index.bincount(minlength=graph.num_node)
    bc_order = (b_index * graph.num_node + c_index).argsort()
    b_neighbor_c = c_index[bc_order]
    b_rel_bc = rel_bc[bc_order]

    a_index, b_index, rel_ab = graph.edge_list.t()
    ab_b_neighbor_c, ab_degree_b = variadic_index(b_neighbor_c, bc_degree_b, b_index)
    ab_b_rel_bc = variadic_index(b_rel_bc, bc_degree_b, b_index)[0]
    ab_bc_a = a_index.repeat_interleave(ab_degree_b)
    ab_bc_rel_ab = rel_ab.repeat_interleave(ab_degree_b)
    # print("per 2-hop random walk variable: %g GiB" % (ab_bc_a.numel() * ab_bc_a.element_size() / 1024 ** 3))
    body = tuple2scalar((ab_bc_rel_ab, ab_b_rel_bc), (graph.num_relation, graph.num_relation))
    # mask = body2id.has_key(body)
    mask = body2id[body] >= 0
    # deduplicate the body groundings w.r.t. a_index & c_index
    # each node pair (a, c) can at most have one grounding for a given rule body
    grounding = tuple2scalar((ab_bc_a[mask], ab_b_neighbor_c[mask], ab_bc_rel_ab[mask], ab_b_rel_bc[mask]),
                             (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))
    grounding = grounding.unique()
    del a_index, b_index, c_index, rel_ab, rel_bc, bc_degree_b, bc_order, b_neighbor_c, \
        b_rel_bc, ab_b_neighbor_c, ab_degree_b, ab_b_rel_bc, ab_bc_a, ab_bc_rel_ab, body, mask

    # enumerate all 1-hop random walks a -> c'
    a_index, c_index, rel_ac = graph.edge_list.t()
    # deduplicate the head groundings w.r.t. a_index / c_index
    a_head = tuple2scalar((a_index, rel_ac), (graph.num_node, graph.num_relation))
    c_head = tuple2scalar((c_index, rel_ac), (graph.num_node, graph.num_relation))
    a_head = a_head.unique()
    c_head = c_head.unique()
    a_index = scalar2tuple(a_head, (graph.num_node, graph.num_relation))[0]
    a_num_relation = a_index.bincount(minlength=graph.num_node)
    c_index = scalar2tuple(c_head, (graph.num_node, graph.num_relation))[0]
    c_num_relation = c_index.bincount(minlength=graph.num_node)
    del a_index, c_index, rel_ac

    # broadcast a -> b -> c to a -> c'
    # c and c' may be different nodes
    a_index = scalar2tuple(grounding, (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))[0]
    # shard the evidences to avoid OOM
    num_cum_evidence = a_num_relation[a_index].cumsum(0)
    num_evidence = num_cum_evidence[-1].item()
    num_grounding = len(grounding)
    # every shard is no more than num_grounding elements
    num_shard = num_evidence // num_grounding + int(num_evidence % num_grounding > 0)
    # print("#shard: %d" % num_shard)
    keys = torch.arange(1, num_shard + 1, device=graph.device) * num_grounding
    right = torch.bucketize(keys, num_cum_evidence, right=True)
    num_a_evidence = torch.zeros(len(rule_set), dtype=torch.long, device=graph.device)
    last = 0
    for current in right.tolist():
        mask = slice(last, current)
        a_body = grounding[mask].repeat_interleave(a_num_relation[a_index[mask]])
        rel_ac = scalar2tuple(a_head, (graph.num_node, graph.num_relation))[1]
        rel_ac = variadic_index(rel_ac, a_num_relation, a_index[mask])[0]
        # rel_ab, rel_bc = \
        #     scalar2tuple(a_body, (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))[2:]
        # a_evidence = tuple2scalar((rel_ab, rel_bc, rel_ac), (graph.num_relation, graph.num_relation, graph.num_relation))
        a_evidence = a_body % (graph.num_relation ** 2) * graph.num_relation + rel_ac
        # a_evidence = a_evidence[rule2id.has_key(a_evidence)]
        a_evidence = a_evidence[rule2id[a_evidence] >= 0]
        # num_a_evidence += rule2id[a_evidence].bincount(minlength=len(rule_set))  # too slow
        ones = torch.ones(len(a_evidence), dtype=torch.int, device=graph.device)
        num_a_evidence += scatter_add(ones, rule2id[a_evidence], dim_size=len(rule_set)).long()  # fast
        last = current
    del a_index, num_cum_evidence, keys, right, mask, a_body, rel_ac, a_evidence, ones

    # broadcast a -> b -> c to a' -> c
    # a and a' may be different nodes
    c_index = scalar2tuple(grounding, (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))[1]
    # shard the evidences to avoid OOM
    num_cum_evidence = c_num_relation[c_index].cumsum(0)
    num_evidence = num_cum_evidence[-1].item()
    num_grounding = len(grounding)
    # every shard is no more than num_grounding elements
    num_shard = num_evidence // num_grounding + int(num_evidence % num_grounding > 0)
    # print("#shard: %d" % num_shard)
    keys = torch.arange(1, num_shard + 1, device=graph.device) * num_grounding
    right = torch.bucketize(keys, num_cum_evidence, right=True)
    num_c_evidence = torch.zeros(len(rule_set), dtype=torch.long, device=graph.device)
    last = 0
    for current in right.tolist():
        mask = slice(last, current)
        c_body = grounding[mask].repeat_interleave(c_num_relation[c_index[mask]])
        rel_ac = scalar2tuple(c_head, (graph.num_node, graph.num_relation))[1]
        rel_ac = variadic_index(rel_ac, c_num_relation, c_index[mask])[0]
        # rel_ab, rel_bc = \
        #     scalar2tuple(c_body, (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))[2:]
        # c_evidence = tuple2scalar((rel_ab, rel_bc, rel_ac), (graph.num_relation, graph.num_relation, graph.num_relation))
        c_evidence = c_body % (graph.num_relation ** 2) * graph.num_relation + rel_ac
        # c_evidence = c_evidence[rule2id.has_key(c_evidence)]
        c_evidence = c_evidence[rule2id[c_evidence] >= 0]
        # num_c_evidence += rule2id[c_evidence].bincount(minlength=len(rule_set))  # too slow
        ones = torch.ones(len(c_evidence), dtype=torch.int, device=graph.device)
        num_c_evidence += scatter_add(ones, rule2id[c_evidence], dim_size=len(rule_set)).long()  # fast
        last = current
    del c_index, num_cum_evidence, keys, right, mask, c_body, rel_ac, c_evidence, ones

    num_evidence = (num_a_evidence + num_c_evidence) / 2
    confidence = num_support / num_evidence

    mask = (coverage >= min_coverage) & (confidence >= min_confidence)
    rel_ab, rel_bc, rel_ac = scalar2tuple(rule_set[mask], (graph.num_relation, graph.num_relation, graph.num_relation))
    coverage = coverage[mask]
    confidence = confidence[mask]

    return rel_ab, rel_bc, rel_ac, coverage, confidence


def rule_mining_naive(graph, min_coverage=0.1, min_confidence=0.1):
    from tqdm import tqdm

    edge_list = graph.edge_list.tolist()
    h2trs = defaultdict(list)
    t2hrs = defaultdict(list)
    htrs = []
    ht2rs = defaultdict(list)
    h2rs = defaultdict(set)
    t2rs = defaultdict(set)
    num_r = defaultdict(int)
    for h, t, r in edge_list:
        h2trs[h].append((t, r))
        t2hrs[t].append((h, r))
        htrs.append((h, t, r))
        ht2rs[(h, t)].append(r)
        h2rs[h].add(r)
        t2rs[t].add(r)
        num_r[r] += 1

    body2heads = defaultdict(set)
    num_support = defaultdict(int)
    visited = set()
    for a in tqdm(h2trs):
        for b, rel_ab in h2trs[a]:
            for c, rel_bc in h2trs[b]:
                if (a, c) not in ht2rs:
                    continue
                if (a, c, rel_ab, rel_bc) in visited:
                    continue
                visited.add((a, c, rel_ab, rel_bc))
                for rel_ac in ht2rs[(a, c)]:
                    body2heads[(rel_ab, rel_bc)].add(rel_ac)
                    num_support[(rel_ab, rel_bc, rel_ac)] += 1

    coverage = {}
    for rel_ab, rel_bc, rel_ac in num_support:
        coverage[(rel_ab, rel_bc, rel_ac)] = num_support[(rel_ab, rel_bc, rel_ac)] / num_r[rel_ac]

    num_a_evidence = defaultdict(int)
    num_c_evidence = defaultdict(int)
    visited = set()
    for a in tqdm(h2trs):
        for b, rel_ab in h2trs[a]:
            for c, rel_bc in h2trs[b]:
                if (rel_ab, rel_bc) not in body2heads:
                    continue
                if (a, c, rel_ab, rel_bc) in visited:
                    continue
                visited.add((a, c, rel_ab, rel_bc))
                for rel_ac1 in h2rs[a].intersection(body2heads[(rel_ab, rel_bc)]):
                    num_a_evidence[(rel_ab, rel_bc, rel_ac1)] += 1
                for rel_a1c in t2rs[c].intersection(body2heads[(rel_ab, rel_bc)]):
                    num_c_evidence[(rel_ab, rel_bc, rel_a1c)] += 1

    confidence = {}
    for rule in num_support:
        confidence[rule] = num_support[rule] / (num_a_evidence[rule] + num_c_evidence[rule]) * 2
    rule = [r for r in num_support if coverage[r] >= min_coverage and confidence[r] >= min_confidence]
    confidence = [confidence[r] for r in rule]
    coverage = [coverage[r] for r in rule]
    rel_ab, rel_bc, rel_ac = zip(*rule)

    return rel_ab, rel_bc, rel_ac, coverage, confidence


def rule_inference(graph, rel_ab, rel_bc, rel_ac):
    body = tuple2scalar((rel_ab, rel_bc), (graph.num_relation, graph.num_relation))
    body_set = body.unique()
    rule = tuple2scalar((rel_ab, rel_bc, rel_ac), (graph.num_relation, graph.num_relation, graph.num_relation))
    rule_set = rule.unique()
    # discretize body_set
    body2id = -torch.ones(graph.num_relation ** 2, dtype=torch.long, device=graph.device)  # fast, but cost memory
    body2id[body_set] = torch.arange(len(body_set), device=graph.device)
    body_rule = rule_set
    body = rule // graph.num_relation
    ones = torch.ones(len(body), dtype=torch.int, device=graph.device)
    body_num_rule = scatter_add(ones, body2id[body], dim_size=len(body_set)).long()

    # enumerate all 2-hop random walks a -> b -> c
    b_index, c_index, rel_bc = graph.edge_list.t()
    bc_degree_b = b_index.bincount(minlength=graph.num_node)
    bc_order = (b_index * graph.num_node + c_index).argsort()
    b_neighbor_c = c_index[bc_order]
    b_rel_bc = rel_bc[bc_order]

    a_index, b_index, rel_ab = graph.edge_list.t()
    ab_b_neighbor_c, ab_degree_b = variadic_index(b_neighbor_c, bc_degree_b, b_index)
    ab_b_rel_bc = variadic_index(b_rel_bc, bc_degree_b, b_index)[0]
    ab_bc_a = a_index.repeat_interleave(ab_degree_b)
    ab_bc_rel_ab = rel_ab.repeat_interleave(ab_degree_b)
    body = tuple2scalar((ab_bc_rel_ab, ab_b_rel_bc), (graph.num_relation, graph.num_relation))
    mask = body2id[body] >= 0
    # deduplicate the body groundings w.r.t. a_index & c_index
    # each node pair (a, c) can at most have one grounding for a given rule body
    grounding = tuple2scalar((ab_bc_a[mask], ab_b_neighbor_c[mask], ab_bc_rel_ab[mask], ab_b_rel_bc[mask]),
                             (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))
    grounding = grounding.unique()

    a_index, c_index, rel_ab, rel_bc = \
        scalar2tuple(grounding, (graph.num_node, graph.num_node, graph.num_relation, graph.num_relation))
    body = grounding % (graph.num_relation ** 2)
    a_index = a_index.repeat_interleave(body_num_rule[body2id[body]])
    c_index = c_index.repeat_interleave(body_num_rule[body2id[body]])
    ac_rule = variadic_index(body_rule, body_num_rule, body2id[body])[0]
    rel_ac = ac_rule % graph.num_relation

    prediction = tuple2scalar((a_index, c_index, rel_ac), (graph.num_node, graph.num_node, graph.num_relation))
    prediction = prediction.unique()
    a_index, c_index, rel_ac = scalar2tuple(prediction, (graph.num_node, graph.num_node, graph.num_relation))

    return a_index, c_index, rel_ac


def single_source_triangle_list(graph, edge_ab=None, edge_bc=None, edge_ac=None, max_triangle_per_edge=None):
    # edge_ac must have unique c nodes
    assert edge_ac is not None

    a_index = graph.edge_list[edge_ac, 0].unique()
    if edge_ab is None:
        edge_ab = graph.edge_list[:, 0] == a_index.repeat_interleave(graph.num_edges)
    else:
        if edge_ab.dtype == torch.long:
            edge_ab = functional.as_mask(edge_ab, graph.num_edge)
        edge_ab = edge_ab & (graph.edge_list[:, 0] == a_index.repeat_interleave(graph.num_edges))
    edge_ab = edge_ab.nonzero().squeeze(-1)
    if edge_bc is None:
        edge_bc = torch.arange(graph.num_edge, device=graph.device)
    elif edge_bc.dtype == torch.bool:
        edge_bc = edge_bc.nonzero().squeeze(-1)
    if edge_ac.dtype == torch.bool:
        edge_ac = edge_ac.nonzero().squeeze(-1)

    # find all triangles that satisfy a -> c and a -> b -> c
    a_index, b_index = graph.edge_list[edge_ab, :2].t()
    ab_order = b_index.argsort()
    a_neighbor_b = b_index[ab_order]

    a_index, c_index = graph.edge_list[edge_ac, :2].t()
    ac_order = c_index.argsort()
    a_neighbor_c = c_index[ac_order]
    range = torch.arange(len(graph), device=graph.device)
    assert (graph.node2graph[a_index.unique()] == range).all()

    b_index, c_index = graph.edge_list[edge_bc, :2].t()
    bc_order = c_index.argsort()
    b_index = b_index[bc_order]
    c_index = c_index[bc_order]
    bc_degree_c = segment_add_coo(torch.ones_like(c_index), c_index, dim_size=graph.num_node)  # fast
    # bc_degree_c = c_index.bincount(minlength=graph.num_node)  # too slow due to atomic
    b_left = torch.bucketize(b_index, a_neighbor_b)
    b_right = torch.bucketize(b_index, a_neighbor_b, right=True)
    b_count = b_right - b_left
    c_left = torch.bucketize(c_index, a_neighbor_c)
    c_right = torch.bucketize(c_index, a_neighbor_c, right=True)
    c_count = c_right - c_left
    assert (c_count <= 1).all()
    count = b_count * c_count
    count_per_ac = variadic_sum(count, bc_degree_c)
    sample_per_ac = count_per_ac
    if max_triangle_per_edge is not None:
        sample_per_ac = sample_per_ac.clamp(max=max_triangle_per_edge)
        rand = torch.rand(sample_per_ac.sum(), device=graph.device)
        randint = (rand * count_per_ac.repeat_interleave(sample_per_ac)).long()
    else:
        randint = functional.variadic_arange(sample_per_ac)

    offset = count_per_ac.max()
    cum_count_shifted = (count.cumsum(0) - count) - \
                        (count_per_ac.cumsum(0) - count_per_ac).repeat_interleave(bc_degree_c)
    assert (cum_count_shifted <= offset).all()
    assert (randint < offset).all()
    boundary = cum_count_shifted + offset * functional._size_to_index(bc_degree_c)
    key = randint + offset * functional._size_to_index(sample_per_ac)
    index = torch.bucketize(key, boundary, right=True) - 1

    assert (b_count[index] >= 1).all()
    assert (c_count[index] >= 1).all()
    edge_ab = edge_ab[ab_order[key - boundary[index] + b_left[index]]]
    edge_bc = edge_bc[bc_order[index]]
    edge_ac = edge_ac[ac_order[c_left[index]]]

    # sanity check
    # check_triangles(graph, edge_ab, edge_bc, edge_ac)

    return edge_ab, edge_bc, edge_ac