from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, segment_add_coo

from torchdrug import core, data, layers, utils
from torchdrug.utils import comm
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import layer


@R.register("model.LogicNN")
class LogicMessagePassingNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, max_triangle=None, message_func="distmult",
                 aggregate_func="sum", short_cut=False, layer_norm=False, activation="relu", concat_hidden=False,
                 num_mlp_layer=2, remove_one_hop=False, independent_fact=False, dependent=False, only_init_loop=False,
                 dependent_fact_init=False, dependent_aux_init=False, no_aux_relation_type=False, zero_aux_init=False,
                 relation_as_fact=False, precompute_degree=False, pre_activation=False, self_loop=False,
                 readout_unary=False, min_coverage=0, min_confidence=0, message_trans=False, auxiliary_filter=False,
                 rule_filter=False, fff_update=False, auxiliary_node=False):
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
        self.dependent_fact_init = dependent_fact_init
        self.dependent_aux_init = dependent_aux_init
        self.no_aux_relation_type = no_aux_relation_type
        self.zero_aux_init = zero_aux_init
        self.relation_as_fact = relation_as_fact
        self.precompute_degree = precompute_degree
        self.self_loop = self_loop
        self.readout_unary = readout_unary
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.auxiliary_filter = auxiliary_filter
        self.rule_filter = rule_filter
        self.fff_update = fff_update
        self.auxiliary_node = auxiliary_node

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.LogicMessagePassingConv(self.dims[i], self.dims[i + 1], num_relation * 2,
                                                             message_func, aggregate_func, layer_norm, activation,
                                                             dependent, pre_activation, message_trans))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(num_relation * 2, input_dim)
        if independent_fact:
            self.layer_facts = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_facts.append(nn.Embedding(num_relation * 2, self.dims[i]))
        if dependent_fact_init:
            self.fact_linear = nn.Linear(input_dim, num_relation * 2 * input_dim)
        else:
            self.fact = nn.Embedding(num_relation * 2, input_dim)
        if dependent_aux_init:
            self.aux_linear = nn.Linear(input_dim, num_relation * 2 * input_dim)
        else:
            self.aux = nn.Embedding(num_relation * 2, input_dim)
        if readout_unary:
            feature_dim = feature_dim * 3 - input_dim * 2
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
        # only mask fact edges
        edge_mask = edge_mask | graph.is_auxiliary | graph.is_loop

        if hasattr(graph, "is_auxiliary"):
            ab_aux = rule_inference(graph, *self.rules, edge_ab=~edge_mask)
            bc_aux = rule_inference(graph, *self.rules, edge_bc=~edge_mask)
            a_index, c_index, rel_ac = zip(ab_aux, bc_aux)
            a_index = torch.cat(a_index)
            c_index = torch.cat(c_index)
            rel_ac = torch.cat(rel_ac)
            pattern = torch.stack([a_index, c_index, rel_ac], dim=-1)
            edge_index = graph.match(pattern)[0]
            aux_mask = ~(functional.as_mask(edge_index, graph.num_edge) & graph.is_auxiliary)
            edge_mask = edge_mask & aux_mask

        graph = graph.edge_mask(edge_mask)

        if hasattr(graph, "edge_ab"):
            mask = (graph.edge_ab >= 0) & (graph.edge_bc >= 0) & (graph.edge_ac >= 0)
            with graph.edge_reference():
                graph.edge_ab = graph.edge_ab[mask]
                graph.edge_bc = graph.edge_bc[mask]
                graph.edge_ac = graph.edge_ac[mask]

        return graph

    def add_auxiliary(self, graph, rel_ab, rel_bc, rel_ac, auxiliary_filter=False, rule_filter=False, h_index=None, t_index=None, r_index=None):
        assert not isinstance(graph, data.PackedGraph)

        if rule_filter:
            batch_size = len(r_index)
            q_r = r_index[:, 0]
            q_r = torch.cat([q_r, q_r + graph.num_relation])
            rel_ab_batch = rel_ab.repeat(batch_size * 2, 1)
            rel_bc_batch = rel_bc.repeat(batch_size * 2, 1)
            rel_ac_batch = rel_ac.repeat(batch_size * 2, 1)
            rel_ab_mask = rel_ab_batch == q_r.unsqueeze(1)
            rel_bc_mask = rel_bc_batch == q_r.unsqueeze(1)
            rel_ac_mask = rel_ac_batch == q_r.unsqueeze(1)
            rel_ab_batch_mask = rel_ab_mask.sum(dim=0)
            rel_bc_batch_mask = rel_bc_mask.sum(dim=0)
            rel_ac_batch_mask = rel_ac_mask.sum(dim=0)
            rule_mask = (rel_ab_batch_mask == 1) | (rel_bc_batch_mask == 1) | (rel_ac_batch_mask == 1)
            assert rule_mask.shape == rel_ab.shape
            print('original rule number: ', rel_ab.shape)
            rel_ab = rel_ab[rule_mask]
            print('filtered rule number: ', rel_ab.shape)
            rel_bc = rel_bc[rule_mask]
            rel_ac = rel_ac[rule_mask]
        a_index, c_index, rel_ac = rule_inference(graph, rel_ab, rel_bc, rel_ac)
        if auxiliary_filter:
            assert h_index is not None
            assert t_index is not None
            is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
            new_h_index = torch.where(is_t_neg, h_index, t_index)
            batch_size = len(h_index)
            query = new_h_index[:, 0]
            a_batch = a_index.repeat(batch_size, 1)
            c_batch = c_index.repeat(batch_size, 1)
            a_mask = a_batch == query.unsqueeze(1)
            c_mask = c_batch == query.unsqueeze(1)
            a_batch_mask = sum(a_mask)
            c_batch_mask = sum(c_mask)
            query_related_mask = (a_batch_mask == 1) | (c_batch_mask == 1)
            assert query_related_mask.shape == a_index.shape
            print('original auxiliary number: ', len(a_index))
            a_index = a_index[query_related_mask]
            print('filterted auxiliary number: ', len(a_index))
            c_index = c_index[query_related_mask]
            rel_ac = rel_ac[query_related_mask]
        edge_list = torch.stack([a_index, c_index, rel_ac], dim=-1)
        num_match = graph.match(edge_list)[1]
        mask = num_match == 0
        # only add auxiliary variables that are not in the graph
        edge_list = edge_list[mask]

        edge_weight = torch.ones(len(edge_list), device=self.device)
        not_auxiliary = torch.zeros(graph.num_edge, dtype=torch.bool, device=self.device)
        is_auxiliary = torch.ones(len(edge_list), dtype=torch.bool, device=self.device)
        edge_list = torch.cat([graph.edge_list, edge_list])
        edge_weight = torch.cat([graph.edge_weight, edge_weight])
        is_auxiliary = torch.cat([not_auxiliary, is_auxiliary])
        data_dict, meta_dict = graph.data_by_meta(exclude="edge")
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=graph.num_relation, meta_dict=meta_dict, **data_dict)

        if is_auxiliary.any():
            # fact + fact -> auxiliary
            max_ffa = self.max_triangle.get("ffa", None)
            ffa = triangle_list(graph, edge_ab=~is_auxiliary, edge_bc=~is_auxiliary, edge_ac=is_auxiliary,
                                max_triangle_per_edge=max_ffa)
            # auxiliary + fact -> auxiliary
            max_afa = self.max_triangle.get("afa", None)
            afa = triangle_list(graph, edge_ab=is_auxiliary, edge_bc=~is_auxiliary, edge_ac=is_auxiliary,
                                max_triangle_per_edge=max_afa)
            # fact + auxiliary -> auxiliary
            max_faa = self.max_triangle.get("faa", None)
            faa = triangle_list(graph, edge_ab=~is_auxiliary, edge_bc=is_auxiliary, edge_ac=is_auxiliary,
                                max_triangle_per_edge=max_faa)
            # auxiliary + auxiliary -> auxiliary
            max_aaa = self.max_triangle.get("aaa", None)
            aaa = triangle_list(graph, edge_ab=is_auxiliary, edge_bc=is_auxiliary, edge_ac=is_auxiliary,
                                max_triangle_per_edge=max_aaa)
            edge_ab, edge_bc, edge_ac = zip(ffa, afa, faa, aaa)
            with graph.edge_reference():
                graph.edge_ab = torch.cat(edge_ab)
                graph.edge_bc = torch.cat(edge_bc)
                graph.edge_ac = torch.cat(edge_ac)

        with graph.edge():
            graph.is_auxiliary = is_auxiliary

        return graph

    def add_self_loop(self, graph):
        h_index = t_index = torch.arange(graph.num_node, device=self.device)
        r_index = torch.zeros(graph.num_node, dtype=torch.long, device=self.device)
        edge_list = torch.stack([h_index, t_index, r_index], dim=-1)

        edge_weight = torch.ones(len(edge_list), device=self.device)
        not_auxiliary = torch.zeros(len(edge_list), dtype=torch.bool, device=self.device)
        not_loop = torch.zeros(graph.num_edge, dtype=torch.bool, device=self.device)
        is_loop = torch.ones(len(edge_list), dtype=torch.bool, device=self.device)
        edge_list = torch.cat([graph.edge_list, edge_list])
        edge_weight = torch.cat([graph.edge_weight, edge_weight])
        is_loop = torch.cat([not_loop, is_loop])
        is_auxiliary = torch.cat([graph.is_auxiliary, not_auxiliary])
        data_dict, meta_dict = graph.data_by_meta(exclude="edge")
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=graph.num_relation, meta_dict=meta_dict, **data_dict)

        if hasattr(graph, "edge_ab"):
            # no need to remap edges as we only append new edges for a single graph
            triangles = [(graph.edge_ab, graph.edge_bc, graph.edge_ac)]
        else:
            triangles = []
        edge_bc = (~is_auxiliary & ~is_loop).nonzero().squeeze(-1)
        h_index, t_index = graph.edge_list[edge_bc, :2].t()
        edge_ab = graph.num_edge - graph.num_node + h_index
        edge_ac = graph.num_edge - graph.num_node + t_index
        assert (graph.edge_list[edge_ab, :2] == h_index.unsqueeze(-1)).all()
        assert (graph.edge_list[edge_ac, :2] == t_index.unsqueeze(-1)).all()
        triangles += [(edge_ab, edge_bc, edge_ac)]

        edge_ab, edge_bc, edge_ac = zip(*triangles)
        with graph.edge_reference():
            graph.edge_ab = torch.cat(edge_ab)
            graph.edge_bc = torch.cat(edge_bc)
            graph.edge_ac = torch.cat(edge_ac)
        with graph.edge():
            graph.is_auxiliary = is_auxiliary
            graph.is_loop = is_loop

        return graph

    def add_query_specific(self, graph, h_index, t_index, r_index):
        if not self.auxiliary_node:
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
            not_auxiliary = torch.zeros(len(edge_list), dtype=torch.bool, device=self.device)
            not_loop = torch.zeros(len(edge_list), dtype=torch.bool, device=self.device)
            num_query_specific = torch.ones(len(h_index), dtype=torch.long, device=self.device) * graph.num_nodes[0]

            edge_list, num_edges = functional._extend(graph.edge_list, graph.num_edges, edge_list, num_query_specific)
            edge_weight = functional._extend(graph.edge_weight, graph.num_edges, edge_weight, num_query_specific)[0]
            is_auxiliary = functional._extend(graph.is_auxiliary, graph.num_edges, not_auxiliary, num_query_specific)[0]
            is_loop = functional._extend(graph.is_loop, graph.num_edges, not_loop, num_query_specific)[0]
            is_query = functional._extend(not_query, graph.num_edges, is_query, num_query_specific)[0]
            start = num_edges.cumsum(0) - num_edges
            end = start + graph.num_edges
            mask = functional.multi_slice_mask(start, end, num_edges.sum())
            data_dict, meta_dict = graph.data_by_meta(exclude="edge")
            offsets = graph._get_offsets(graph.num_nodes, num_edges)
            graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                                num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)
        else:
            all_index = torch.arange(graph.num_nodes[0] - 1, device=self.device)
            offset = graph.num_cum_nodes - graph.num_nodes
            aux_node = offset + graph.num_nodes - 1
            aux_index, t_index = torch.meshgrid(aux_node, all_index)
            t_index = t_index + offset.unsqueeze(-1)
            r_index = r_index[:, [0]].expand_as(aux_index)
            edge_list = torch.stack([aux_index, t_index, r_index], dim=-1)
            edge_list = edge_list.flatten(0, -2)

            edge_weight = torch.ones(len(edge_list), device=self.device)
            not_query = torch.zeros(graph.num_edge, dtype=torch.bool, device=self.device)
            not_auxiliary = torch.zeros(len(edge_list), dtype=torch.bool, device=self.device)
            not_loop = torch.zeros(len(edge_list), dtype=torch.bool, device=self.device)
            is_query = torch.ones(len(edge_list), dtype=torch.bool, device=self.device)

            num_new_edges = torch.ones(len(aux_index), dtype=torch.long, device=self.device) * (graph.num_nodes - 1)
            edge_list, num_edges = functional._extend(graph.edge_list, graph.num_edges, edge_list, num_new_edges)
            edge_weight = functional._extend(graph.edge_weight, graph.num_edges, edge_weight, num_new_edges)[0]
            is_auxiliary = functional._extend(graph.is_auxiliary, graph.num_edges, not_auxiliary, num_new_edges)[0]
            is_loop = functional._extend(graph.is_loop, graph.num_edges, not_loop, num_new_edges)[0]
            is_query = functional._extend(not_query, graph.num_edges, is_query, num_new_edges)[0]

            start = num_edges.cumsum(0) - num_edges
            end = start + graph.num_edges
            mask = functional.multi_slice_mask(start, end, num_edges.sum())
            data_dict, meta_dict = graph.data_by_meta(exclude="edge")
            offsets = graph._get_offsets(graph.num_nodes, num_edges)
            graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                                num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict,
                                **data_dict)

        if hasattr(graph, "edge_ab"):
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            triangles = [(mapping[graph.edge_ab], mapping[graph.edge_bc], mapping[graph.edge_ac])]
        else:
            triangles = []
        # fact / aux + fact / aux -> query
        max_ffq = self.max_triangle.get("ffq", None)
        ffq = single_source_triangle_list(graph, edge_ab=~is_query, edge_bc=~is_query, edge_ac=is_query,
                                          max_triangle_per_edge=max_ffq)
        # query + fact / aux -> query
        max_qfq = self.max_triangle.get("qfq", None)
        qfq = single_source_triangle_list(graph, edge_ab=is_query, edge_bc=~is_query & ~is_loop, edge_ac=is_query,
                                          max_triangle_per_edge=max_qfq)
        # query + loop -> query
        max_qsq = self.max_triangle.get("qsq", None)
        qsq = single_source_triangle_list(graph, edge_ab=is_query, edge_bc=is_loop, edge_ac=is_query,
                                          max_triangle_per_edge=max_qsq)
        # query + query -> query
        max_qqq = self.max_triangle.get("qqq", None)
        qqq = single_source_triangle_list(graph, edge_ab=is_query, edge_bc=is_query, edge_ac=is_query,
                                          max_triangle_per_edge=max_qqq)
        triangles += [ffq, qfq, qsq, qqq]
        edge_ab, edge_bc, edge_ac = zip(*triangles)

        with graph.edge():
            graph.is_auxiliary = is_auxiliary
            graph.is_loop = is_loop
            graph.is_query = is_query
        with graph.edge_reference():
            graph.edge_ab = torch.cat(edge_ab)
            graph.edge_bc = torch.cat(edge_bc)
            graph.edge_ac = torch.cat(edge_ac)
        with graph.graph():
            graph.query = r_index[:, 0]

        return graph

    def relation_as_fact_graph(self, graph, fff_update=False, rel_ab=None, rel_bc=None, rel_ac=None):
        batch_size = len(graph)
        relation = torch.arange(graph.num_relation, device=self.device).repeat(batch_size)
        entity = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(graph.num_relation)
        edge_list = torch.stack([entity, entity, relation], dim=-1)
        edge_weight = torch.ones(len(relation), device=self.device)
        num_edges = torch.ones(batch_size, dtype=torch.long, device=self.device) * graph.num_relation
        is_variable = graph.is_query | graph.is_auxiliary | graph.is_loop
        var_list = graph.edge_list[is_variable]
        var_weight = graph.edge_weight[is_variable]
        var2graph = graph.edge2graph[is_variable]
        num_vars = segment_add_coo(torch.ones_like(var2graph), var2graph, dim_size=batch_size)
        not_var = torch.zeros(num_edges.sum(), dtype=torch.bool, device=self.device)
        edge_list = functional._extend(edge_list, num_edges, var_list, num_vars)[0]
        edge_weight = functional._extend(edge_weight, num_edges, var_weight, num_vars)[0]
        is_auxiliary = functional._extend(not_var, num_edges, graph.is_auxiliary[is_variable], num_vars)[0]
        is_loop = functional._extend(not_var, num_edges, graph.is_loop[is_variable], num_vars)[0]
        is_query, num_edges = functional._extend(not_var, num_edges, graph.is_query[is_variable], num_vars)
        num_cum_edges = num_edges.cumsum(0)

        relation = graph.edge_list[:, 2]
        mapping = torch.arange(graph.num_edge, device=self.device)
        mapping[is_variable] = mapping[is_variable] + \
                               (num_cum_edges - graph.num_cum_edges).repeat_interleave(num_vars)
        mapping[~is_variable] = relation[~is_variable] + \
                                (num_cum_edges - num_edges).repeat_interleave(graph.num_edges - num_vars)
        data_dict, meta_dict = graph.data_by_meta(exclude="edge")
        offsets = graph._get_offsets(graph.num_nodes, num_edges)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                            num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)

        with graph.edge():
            graph.is_auxiliary = is_auxiliary
            graph.is_loop = is_loop
            graph.is_query = is_query
        with graph.edge_reference():
            graph.edge_ab = mapping[graph.edge_ab]
            graph.edge_bc = mapping[graph.edge_bc]
            graph.edge_ac = mapping[graph.edge_ac]

        if fff_update:
            assert rel_ab is not None
            assert rel_bc is not None
            assert rel_ac is not None
            offset = graph.num_cum_edges - graph.num_edges
            edge_ab = rel_ab + offset.unsqueeze(1)
            edge_bc = rel_bc + offset.unsqueeze(1)
            edge_ac = rel_ac + offset.unsqueeze(1)
            edge_ab = edge_ab.flatten()
            edge_bc = edge_bc.flatten()
            edge_ac = edge_ac.flatten()
            graph.edge_ab = torch.cat([graph.edge_ab, edge_ab])
            graph.edge_bc = torch.cat([graph.edge_bc, edge_bc])
            graph.edge_ac = torch.cat([graph.edge_ac, edge_ac])

        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def get_input_embedding(self, graph, h_index=None, r_index=None):
        relation = graph.edge_list[:, 2]
        with graph.graph():
            graph.query = self.query(graph.query)
        if self.dependent_fact_init:
            fact = self.fact_linear(graph.query).view(-1, self.num_relation * 2, self.dims[0])
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
        else:
            fact = self.fact(relation)
        if self.no_aux_relation_type:
            relation = torch.zeros_like(relation)
        if self.dependent_aux_init:
            aux = self.aux_linear(graph.query).view(-1, self.num_relation * 2, self.dims[0])
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            aux = aux[sample, relation]
        elif self.zero_aux_init:
            aux = torch.zeros(graph.num_edge, self.dims[0], device=self.device)
        else:
            aux = self.aux(relation)

        query = self.query(relation)
        loop = torch.zeros(graph.num_edge, self.dims[0], device=self.device)
        input = torch.where(graph.is_query.unsqueeze(-1), query, fact)
        input = torch.where(graph.is_auxiliary.unsqueeze(-1), aux, input)
        input = torch.where(graph.is_loop.unsqueeze(-1), loop, input)
        if self.auxiliary_node:
            assert self.only_init_loop is True
            _h_index, _t_index = graph.edge_list.t()[:2]
            offsets = graph.num_cum_nodes - graph.num_nodes
            h_index = h_index[:, 0] + offsets
            mask = graph.is_query & (_t_index != h_index[graph.edge2graph])
            input = torch.where(mask.unsqueeze(-1), torch.zeros_like(input), input)
        elif self.only_init_loop:
            h_index, t_index = graph.edge_list.t()[:2]
            mask = graph.is_query & (h_index != t_index)
            input = torch.where(mask.unsqueeze(-1), torch.zeros_like(input), input)
        return input

    def forward_chaining(self, graph, input, all_loss=None, metric=None, graph_fff=None, graph_query=None):
        # check_triangles(graph, graph.edge_ab, graph.edge_bc, graph.edge_ac)
        with graph.edge():
            graph.boundary = input

        if metric is not None:
            ab_type = (graph.is_query[graph.edge_ab] * 2 + graph.is_auxiliary[graph.edge_ab]) * 2 + graph.is_loop[graph.edge_ab]
            bc_type = (graph.is_query[graph.edge_bc] * 2 + graph.is_auxiliary[graph.edge_bc]) * 2 + graph.is_loop[graph.edge_bc]
            ac_type = (graph.is_query[graph.edge_ac] * 2 + graph.is_auxiliary[graph.edge_ac]) * 2 + graph.is_loop[graph.edge_ac]
            triangle_type = (ab_type * 8 + bc_type) * 8 + ac_type
            ones = torch.ones(len(triangle_type), dtype=torch.int, device=self.device)
            num_triangle = scatter_add(ones, triangle_type, dim_size=512).long()
            # collect triangles from all devices to avoid missing any triangle type
            if comm.get_world_size() > 1:
                num_triangle_all = comm.reduce(num_triangle)
            else:
                num_triangle_all = num_triangle
            metric["#triangle"] = num_triangle.sum() / len(graph)
            triangle_type = num_triangle_all.nonzero().flatten()
            type_name = {0: "fact", 1: "loop", 2: "aux", 4: "query"}
            for t in triangle_type.tolist():
                name = "#%s + %s -> %s" % (type_name[(t & 0x1C0) // 64], type_name[(t & 0x38) // 8], type_name[t & 0x7])
                metric[name] = num_triangle[t] / len(graph)

            variable_type = (graph.is_query * 2 + graph.is_auxiliary) * 2 + graph.is_loop
            ones = torch.ones(len(variable_type), dtype=torch.int, device=self.device)
            num_variable = scatter_add(ones, variable_type, dim_size=8).long()
            metric["#variable"] = num_variable.sum() / len(graph)
            for t in type_name:
                metric["#%s" % type_name[t]] = num_variable[t] / len(graph)

        hiddens = []
        layer_input = input

        for i, layer in enumerate(self.layers):
            if self.independent_fact:
                relation = graph.edge_list[:, 2]
                layer_fact = self.layer_facts[i](relation)
                layer_input = torch.where(graph.is_query.unsqueeze(-1), layer_input, layer_fact)
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        query = graph.query.repeat_interleave(graph.num_edges, dim=0)
        if self.concat_hidden:
            output = torch.cat(hiddens + [query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], query], dim=-1)

        return output

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        assert graph.num_relation == self.num_relation

        if not hasattr(self, "graph") or (self.graph.edge_list != graph.edge_list).any():
            self.graph = graph
            graph = graph.undirected(add_inverse=True)
            rules = rule_mining(graph, self.min_coverage, self.min_confidence)[:3]
            graph = self.add_auxiliary(graph, *rules, self.auxiliary_filter, self.rule_filter, h_index, r_index)
            if self.self_loop:
                assert self.auxiliary_node
                graph = self.add_self_loop(graph)
            else:
                with graph.edge():
                    graph.is_loop = torch.zeros(graph.num_edge, dtype=torch.bool, device=self.device)
            self.undirected_graph = graph
            self.rules = rules

        graph = self.undirected_graph
        degree = graph.degree_out + 1
        if self.auxiliary_node:
            graph = type(graph)(graph.edge_list, edge_weight=graph.edge_weight, num_node=graph.num_node + 1,
                                num_edge=graph.num_edge, num_relation=graph.num_relation,
                                meta_dict=graph.meta_dict, **graph.data_dict)
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

        graph = self.add_query_specific(graph, h_index, t_index, r_index)
        if self.relation_as_fact:
            new_graph = self.relation_as_fact_graph(graph, self.fff_update, *self.rules)
            #assert (new_graph.edge_list[new_graph.edge_ab, 2] == graph.edge_list[graph.edge_ab, 2]).all()
            #assert (new_graph.edge_list[new_graph.edge_bc, 2] == graph.edge_list[graph.edge_bc, 2]).all()
            #assert (new_graph.edge_list[new_graph.edge_ac, 2] == graph.edge_list[graph.edge_ac, 2]).all()
            graph = new_graph
        if self.precompute_degree:
            graph.log_degree = degree.log().mean()
        input = self.get_input_embedding(graph, h_index, r_index)
        output = self.forward_chaining(graph, input, all_loss, metric)
        if self.auxiliary_node:
            index = t_index + (graph.num_cum_edges - graph.num_nodes + 1).unsqueeze(-1)
        else:
            index = t_index + (graph.num_cum_edges - graph.num_nodes).unsqueeze(-1)
        assert graph.is_query[index].all()
        assert (graph.edge_list[index, 1] == t_index + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
        if self.readout_unary:
            assert self.auxiliary_node & self.self_loop
            h_loop = h_index + (graph.num_cum_edges - graph.num_nodes * 2 + 2).unsqueeze(-1)
            t_loop = t_index + (graph.num_cum_edges - graph.num_nodes * 2 + 2).unsqueeze(-1)
            assert graph.is_loop[h_loop].all()
            assert graph.is_loop[t_loop].all()
            assert (graph.edge_list[h_loop, 0] == h_index + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
            assert (graph.edge_list[t_loop, 0] == t_index + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)).all()
            feature = torch.cat([output[index], output[h_loop, :-self.dims[0]], output[t_loop, :-self.dims[0]]], dim=-1)
        else:
            feature = output[index]

        score = self.mlp(feature).squeeze(-1)

        return score.view(h_index.shape)


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 tied_weights=False, concat_hidden=False, num_mlp_layer=2, pre_activation=False, flip_graph=True,
                 relation_transform=False, num_test_layer=None, attend_relation=False, num_latent_relation=0,
                 dropout=0):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.dims = [input_dim] + list(hidden_dims)
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.flip_graph = flip_graph
        self.relation_transform = relation_transform
        self.num_test_layer = num_test_layer
        self.attend_relation = attend_relation
        self.num_latent_relation = num_latent_relation
        if num_test_layer is not None:
            assert tied_weights

        if flip_graph:
            num_relation *= 2

        self.layers = nn.ModuleList()
        self.query_linears = nn.ModuleList()
        self.relation_keys = nn.ModuleList()
        self.latent_relations = nn.ModuleList()
        if tied_weights:
            self.layers += [layer.GeneralizedRelationalConv(self.dims[0], self.dims[1], num_relation,
                                                            self.dims[0], message_func, aggregate_func, layer_norm,
                                                            activation, pre_activation, relation_transform, dropout)] \
                           * len(hidden_dims)
            if attend_relation:
                self.query_linears += [nn.Linear(self.dims[0], self.dims[0])] * len(hidden_dims)
                self.relation_keys += [nn.Embedding(num_relation + num_latent_relation, self.dims[0])] * len(hidden_dims)
                self.latent_relations += [nn.Embedding(num_latent_relation, self.dims[0])] * len(hidden_dims)
        else:
            for i in range(len(hidden_dims)):
                self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   self.dims[0], message_func, aggregate_func,
                                                                   layer_norm, activation, pre_activation,
                                                                   relation_transform, dropout))
                if attend_relation:
                    self.query_linears.append(nn.Linear(self.dims[i], self.dims[i]))
                    self.relation_keys.append(nn.Embedding(num_relation + num_latent_relation, self.dims[i]))
                    self.latent_relations.append(nn.Embedding(num_latent_relation, self.dims[i])) * len(hidden_dims)

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(1, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def bellmanford(self, graph, h_index, separate_grad=False):
        r_index = torch.zeros_like(h_index)
        query = self.query(r_index)
        h_index = h_index + graph.num_cum_nodes - graph.num_nodes
        boundary = torch.zeros(graph.num_node, self.input_dim, device=self.device)
        boundary[h_index] = query
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        layer_input = boundary

        if self.num_test_layer and not self.training:
            layer = self.layers[0]
            query_linear = self.query_linears[0]
            relation_key = self.relation_keys[0]
            latent_relation = self.latent_relations[0]
            dim = self.dims[0]
            for i in range(self.num_test_layer):
                hidden = layer(graph, layer_input)
                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                if self.attend_relation:
                    query = query_linear(hidden)
                    key = relation_key.weight
                    value = torch.cat([layer.relation.weight, latent_relation.weight])
                    weight = torch.einsum("nd, rd -> nr", query, key) / (dim ** 0.5)
                    attention = F.softmax(weight, dim=-1)
                    layer_input = torch.einsum("nr, rd -> nd", attention, value)
                else:
                    layer_input = hidden
        else:
            for i in range(len(self.layers)):
                hidden = self.layers[i](graph, layer_input)
                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                if self.attend_relation:
                    query = self.query_linears[i](hidden)
                    key = self.relation_keys[i].weight
                    value = torch.cat([self.layers[i].relation.weight, self.latent_relations[i].weight])
                    weight = torch.einsum("nd, rd -> nr", query, key) / (self.dims[i] ** 0.5)
                    attention = F.softmax(weight, dim=-1)
                    layer_input = torch.einsum("nr, rd -> nd", attention, value)
                else:
                    layer_input = hidden

        if self.concat_hidden:
            output = torch.cat(hiddens, dim=-1)
        else:
            output = hiddens[-1]

        return output

    def forward(self, graph, query, all_loss=None, metric=None):
        h_index, t_index = query.t()
        if graph.num_relation and self.flip_graph:
            graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index)
        t_index = t_index + graph.num_cum_nodes - graph.num_nodes
        feature = output[t_index]

        return feature


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
        scalar = torch.div(scalar, r, rounding_mode="floor")
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

    node_in, node_out = graph.edge_list[:, :2].t()
    is_loop = node_in == node_out
    edge_ab, edge_bc, edge_ac = triangle_list(graph, edge_ab=~is_loop, edge_bc=~is_loop, edge_ac=~is_loop)
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
    body2id = -torch.ones(graph.num_relation ** 2, dtype=torch.long, device=graph.device)  # fast, but cost memory
    body2id[body_set] = torch.arange(len(body_set), device=graph.device)
    rule2id = -torch.ones(graph.num_relation ** 3, dtype=torch.long, device=graph.device)  # fast, but cost memory
    rule2id[rule_set] = torch.arange(len(rule_set), device=graph.device)
    ones = torch.ones(len(rule), dtype=torch.int, device=graph.device)
    num_support = scatter_add(ones, rule2id[rule], dim_size=len(rule_set)).long()  # fast
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
        a_evidence = a_body % (graph.num_relation ** 2) * graph.num_relation + rel_ac
        a_evidence = a_evidence[rule2id[a_evidence] >= 0]
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
        c_evidence = c_body % (graph.num_relation ** 2) * graph.num_relation + rel_ac
        c_evidence = c_evidence[rule2id[c_evidence] >= 0]
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


def rule_inference(graph, rel_ab, rel_bc, rel_ac, edge_ab=None, edge_bc=None):
    assert graph.num_relation ** 3 <= torch.iinfo(torch.long).max
    assert graph.num_node ** 2 * graph.num_relation <= torch.iinfo(torch.long).max

    if edge_ab is None:
        edge_ab = torch.arange(graph.num_edge, device=graph.device)
    elif edge_ab.dtype == torch.bool:
        edge_ab = edge_ab.nonzero().squeeze(-1)
    if edge_bc is None:
        edge_bc = torch.arange(graph.num_edge, device=graph.device)
    elif edge_bc.dtype == torch.bool:
        edge_bc = edge_bc.nonzero().squeeze(-1)

    body = tuple2scalar((rel_ab, rel_bc), (graph.num_relation, graph.num_relation))
    body_set = body.unique()
    rule = tuple2scalar((rel_ab, rel_bc, rel_ac), (graph.num_relation, graph.num_relation, graph.num_relation))
    rule_set = rule.unique()
    # discretize body_set
    body2id = -torch.ones(graph.num_relation ** 2, dtype=torch.long, device=graph.device)  # fast, but cost memory
    body2id[body_set] = torch.arange(len(body_set), device=graph.device)
    body_rule = rule_set
    body = torch.div(rule, graph.num_relation, rounding_mode="floor")
    ones = torch.ones(len(body), dtype=torch.int, device=graph.device)
    body_num_rule = scatter_add(ones, body2id[body], dim_size=len(body_set)).long()

    # enumerate all 2-hop random walks a -> b -> c
    b_index, c_index, rel_bc = graph.edge_list[edge_bc].t()
    bc_degree_b = b_index.bincount(minlength=graph.num_node)
    bc_order = (b_index * graph.num_node + c_index).argsort()
    b_neighbor_c = c_index[bc_order]
    b_rel_bc = rel_bc[bc_order]

    a_index, b_index, rel_ab = graph.edge_list[edge_ab].t()
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