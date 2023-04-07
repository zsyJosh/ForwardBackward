from collections.abc import Sequence

import torch
from torch import nn
from torch_scatter import scatter_add

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import layer


@R.register("model.LogicNN")
class LogicMessagePassingNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, max_triangle=None, message_func="distmult",
                 aggregate_func="sum", short_cut=False, layer_norm=False, activation="relu", concat_hidden=False,
                 num_mlp_layer=2, remove_one_hop=False, self_loop=False, only_query_triangles=False,
                 with_query_triangles=False, incremental_triangles=True, only_ffq_triangles=False,
                 with_ffq_triangles=False, only_ffq_query_triangles=False, with_ffq_query_triangles=False,
                 dependent=False):
        super(LogicMessagePassingNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.max_triangle = max_triangle
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.self_loop = self_loop
        self.only_query_triangles = only_query_triangles
        self.with_query_triangles = with_query_triangles
        self.only_ffq_triangles = only_ffq_triangles
        self.with_ffq_triangles = with_ffq_triangles
        self.only_ffq_query_triangles = only_ffq_query_triangles
        self.with_ffq_query_triangles = with_ffq_query_triangles
        self.incremental_triangles = incremental_triangles
        self.dependent = dependent

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.LogicMessagePassingConv(self.dims[i], self.dims[i + 1], message_func,
                                                             aggregate_func, layer_norm, activation, self.dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(num_relation * 2, input_dim)
        if self.dependent:
            self.fact_linear = nn.Linear(input_dim, int(num_relation * 2 * input_dim))
        else:
            self.fact = nn.Embedding(num_relation * 2, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index):
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
        # warning: not necessarily unique, though likely
        h_unique = h_index.t()[0]
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
        num_nodes = graph.num_nodes
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                            num_relation=graph.num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)

        # incremental triangle computation
        if self.incremental_triangles:
            if hasattr(graph, "edge_ab"):
                mapping = torch.arange(graph.num_edge, device=self.device)[mask]
                # add triangles: fact + query -> query, fact + fact -> query
                edge_ab, edge_bc, edge_ac = triangle_list(graph, edge_ac=is_query)
                assert graph.edge_ab.max() < len(mapping)
                assert graph.edge_bc.max() < len(mapping)
                assert graph.edge_ac.max() < len(mapping)
                edge_ab = torch.cat([mapping[graph.edge_ab], edge_ab])
                edge_bc = torch.cat([mapping[graph.edge_bc], edge_bc])
                edge_ac = torch.cat([mapping[graph.edge_ac], edge_ac])
                with graph.edge_reference():
                    graph.edge_ab = edge_ab
                    graph.edge_bc = edge_bc
                    graph.edge_ac = edge_ac
        # we may merge the code of only_xxx and with_xxx
        elif self.only_query_triangles:
            edge_ab, edge_bc, edge_ac = self.get_query_triangle(num_nodes, edge_list, h_unique, start, end, self.self_loop)
            check_triangles(graph, edge_ab, edge_bc, edge_ac)
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        elif self.with_query_triangles:
            edge_ab, edge_bc, edge_ac = self.get_query_triangle(num_nodes, edge_list, h_unique, start, end, self.self_loop)
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            assert graph.edge_ab.max() < len(mapping)
            assert graph.edge_bc.max() < len(mapping)
            assert graph.edge_ac.max() < len(mapping)
            edge_ab = torch.cat([mapping[graph.edge_ab], edge_ab])
            edge_bc = torch.cat([mapping[graph.edge_bc], edge_bc])
            edge_ac = torch.cat([mapping[graph.edge_ac], edge_ac])
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        elif self.only_ffq_triangles:
            # add triangles: fact + fact -> query
            edge_ab, edge_bc, edge_ac = self.get_fact_fact_query_triangle(num_nodes, edge_list, h_unique, start, end)
            check_triangles(graph, edge_ab, edge_bc, edge_ac)
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        elif self.with_ffq_triangles:
            edge_ab, edge_bc, edge_ac = self.get_fact_fact_query_triangle(num_nodes, edge_list, h_unique, start, end)
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            assert graph.edge_ab.max() < len(mapping)
            assert graph.edge_bc.max() < len(mapping)
            assert graph.edge_ac.max() < len(mapping)
            edge_ab = torch.cat([mapping[graph.edge_ab], edge_ab])
            edge_bc = torch.cat([mapping[graph.edge_bc], edge_bc])
            edge_ac = torch.cat([mapping[graph.edge_ac], edge_ac])
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        elif self.only_ffq_query_triangles:
            edge_ab1, edge_bc1, edge_ac1 = self.get_fact_fact_query_triangle(num_nodes, edge_list, h_unique, start, end)
            edge_ab2, edge_bc2, edge_ac2 = self.get_query_triangle(num_nodes, edge_list, h_unique, start, end, self.self_loop)
            edge_ab = torch.cat([edge_ab1, edge_ab2])
            edge_bc = torch.cat([edge_bc1, edge_bc2])
            edge_ac = torch.cat([edge_ac1, edge_ac2])
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        elif self.with_ffq_query_triangles:
            edge_ab1, edge_bc1, edge_ac1 = self.get_fact_fact_query_triangle(num_nodes, edge_list, h_unique, start, end)
            edge_ab2, edge_bc2, edge_ac2 = self.get_query_triangle(num_nodes, edge_list, h_unique, start, end,
                                                                   self.self_loop)
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            assert graph.edge_ab.max() < len(mapping)
            assert graph.edge_bc.max() < len(mapping)
            assert graph.edge_ac.max() < len(mapping)
            edge_ab = torch.cat([mapping[graph.edge_ab], edge_ab1, edge_ab2])
            edge_bc = torch.cat([mapping[graph.edge_bc], edge_bc1, edge_bc2])
            edge_ac = torch.cat([mapping[graph.edge_ac], edge_ac1, edge_ac2])
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac
        else:
            # only fact + fact -> fact
            mapping = torch.arange(graph.num_edge, device=self.device)[mask]
            assert graph.edge_ab.max() < len(mapping)
            assert graph.edge_bc.max() < len(mapping)
            assert graph.edge_ac.max() < len(mapping)
            edge_ab = mapping[graph.edge_ab]
            edge_bc = mapping[graph.edge_bc]
            edge_ac = mapping[graph.edge_ac]
            with graph.edge_reference():
                graph.edge_ab = edge_ab
                graph.edge_bc = edge_bc
                graph.edge_ac = edge_ac

        with graph.edge():
            graph.is_query = is_query
        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def get_query_triangle(self, num_nodes, edge_list, h_unique, start, end, self_loop=False):
        edge_ab = []
        edge_bc = []
        edge_ac = []
        # TODO: loop is not parallelized across the batch!
        for i in range(len(start)):
            num_node = num_nodes[i].item()
            query_bucket = torch.arange(num_node, device=num_nodes.device) + end[i]
            query_head = h_unique[i].item()
            query_head_ind = query_bucket[query_head].item()
            # exclude self loop
            query_bucket[query_head] = -1
            query_head += num_nodes[:i].sum()
            all_edges_ind = torch.arange(start[i], end[i], device=num_nodes.device)
            all_edges = edge_list[all_edges_ind]
            all_edges_h, all_edges_t = all_edges.t()
            mask = (all_edges_h != query_head) & (all_edges_t != query_head)
            fact_edges = all_edges[mask]
            fact_edges_h = fact_edges.t()[0] - num_nodes[:i].sum()
            fact_edges_t = fact_edges.t()[1] - num_nodes[:i].sum()
            fact_edges_ind = all_edges_ind[mask]
            # query + fact -> query, where query is not self loop
            edge_bc.append(fact_edges_ind)
            assert (query_bucket[fact_edges_h] > -1).all()
            edge_ab.append(query_bucket[fact_edges_h])
            assert (query_bucket[fact_edges_t] > -1).all()
            edge_ac.append(query_bucket[fact_edges_t])
            if self_loop:
                # query (loop) + fact -> fact
                # TODO: Do query (loop) get updated anywhere?
                self_loop_mask = (all_edges_h == query_head) & (all_edges_t != query_head)
                self_edges = all_edges_ind[self_loop_mask]
                loop_ab = query_head_ind.repeat(len(self_edges))
                edge_ab.append(loop_ab)
                edge_bc.append(self_edges)
                edge_ac.append(self_edges)

        edge_ab = torch.cat(edge_ab, dim=-1)
        edge_bc = torch.cat(edge_bc, dim=-1)
        edge_ac = torch.cat(edge_ac, dim=-1)
        return edge_ab, edge_bc, edge_ac

    def get_fact_fact_query_triangle(self, num_nodes, edge_list, h_unique, start, end):
        edge_ab = []
        edge_bc = []
        edge_ac = []
        # TODO: loop is not parallelized across the batch!
        for i in range(len(start)):
            num_node = num_nodes[i].item()
            # store c's index
            query_bucket = torch.arange(num_node, device=num_nodes.device) + end[i]
            # store b's indices (multi-edges exist)
            query_head = h_unique[i].item()
            query_head += num_nodes[:i].sum()
            all_edges_ind = torch.arange(start[i], end[i], device=num_nodes.device)
            all_edges = edge_list[all_edges_ind]
            all_edges_h = all_edges.t()[0]
            all_edges_t = all_edges.t()[1]
            mask = (all_edges_h == query_head)
            b_candidate = all_edges_t[mask]
            b_candidate_ind = all_edges_ind[mask]
            # recover each fact graph to do graph match (index needs to be shifted!)
            # TODO: this is super expensive as you try to construct perfect hash for every sample!
            fact_graph = data.Graph(all_edges)
            any = -torch.ones_like(b_candidate)
            match_pattern = torch.stack([b_candidate, any, any], dim=-1)
            index, num_match = fact_graph.match(match_pattern)
            index += start[i]
            edge_bc.append(index)

            c_head = edge_list[index].t()[1] - num_nodes[:i].sum()
            ac_ind = query_bucket[c_head]
            edge_ac.append(ac_ind)

            # no need to filter num_match = 0
            # it's okay for repeat_interleave
            ind_mask = (num_match > 0)
            num_matches = num_match[ind_mask]
            b_candidate_ind_select = b_candidate_ind[ind_mask]
            ab_ind = torch.repeat_interleave(b_candidate_ind_select, num_matches)
            assert ab_ind.shape == index.shape
            edge_ab.append(ab_ind)

        edge_ab = torch.cat(edge_ab, dim=-1)
        edge_bc = torch.cat(edge_bc, dim=-1)
        edge_ac = torch.cat(edge_ac, dim=-1)
        return edge_ab, edge_bc, edge_ac

    def get_input_embedding(self, graph, r_index, dependent=False):
        relation = graph.edge_list[:, 2]
        query = self.query(relation)
        if dependent:
            assert (r_index[:, [0]] == r_index).all()
            query_emb = query[r_index[:, [0]]]
            batch_size = r_index.shape[0]
            fact_rel_emb = self.fact_linear(query_emb).view(batch_size, self.num_relation * 2, self.input_dim)
            fact_rel_emb = fact_rel_emb.transpose(0, 1)
            fact = fact_rel_emb[relation]
            query = query.unsqueeze(-2).repeat(1, batch_size, 1)
            assert fact.shape == query.shape
            query_mask = graph.is_query.unsqueeze(-1).unsqueeze(-1).repeat(1, batch_size, 1)
            input = torch.where(query_mask, query, fact)
        else:
            fact = self.fact(relation)
            input = torch.where(graph.is_query.unsqueeze(-1), query, fact)
        return input

    @utils.cached
    def forward_chaining(self, graph, input, all_loss=None, metric=None):
        check_triangles(graph, graph.edge_ab, graph.edge_bc, graph.edge_ac)

        if metric is not None:
            metric["#triangle"] = torch.tensor(len(graph.edge_ab) / len(graph), device=self.device)

        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            output = torch.cat(hiddens + [input], dim=-1)
        else:
            output = torch.cat([hiddens[-1], input], dim=-1)

        return output

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        assert graph.num_relation == self.num_relation * 2
        batch_size = len(h_index)
        graph = graph.repeat(batch_size)
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (graph.num_nodes[0] == graph.num_nodes).all()
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        graph = self.add_query_specific(graph, h_index, t_index, r_index)

        input = self.get_input_embedding(graph, r_index, self.dependent)
        output = self.forward_chaining(graph, input, all_loss, metric)
        index = t_index + (graph.num_cum_edges - graph.num_nodes[0]).unsqueeze(-1)
        assert graph.is_query[index].all()
        if self.dependent:
            num_edges = output.shape[0]
            dim = output.shape[2]
            output = output.transpose(0, 1)
            output_cat = output.reshape(-1, dim)
            ind_shift = torch.arange(0, num_edges * batch_size, step=num_edges, device=h_index.device).unsqueeze(-1)
            index = index + ind_shift
            feature = output_cat[index]
        else:
            feature = output[index]

        score = self.mlp(feature).squeeze(-1)
        return score.view(h_index.shape)


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
    batch_size = len(size)
    count = index.bincount(minlength=batch_size)
    if batch_size > 0 and count.max() > 1:
        # index contains duplicates
        # convert the problem into repeat_interleave + index without duplicates
        input, size = variadic_repeat_interleave(input, size, count)
        index_order = index.argsort()
        index = torch.zeros_like(index)
        index[index_order] = torch.arange(len(index), dtype=torch.long, device=input.device)

    cum_size = size.cumsum(0)
    range = functional.variadic_arange(size[index])
    element_index = range + (cum_size[index] - size[index]).repeat_interleave(size[index])

    return input[element_index], size[index]


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


def triangle_list(graph, edge_ac=None, max_triangle_per_edge=None):
    assert graph.num_node.item() ** 2 <= torch.iinfo(torch.long).max
    assert graph.num_node.item() * graph.num_edge.item() <= torch.iinfo(torch.long).max

    # find all triangles that satisfy a -> c and a -> b -> c
    node_in, node_out = graph.edge_list.t()[:2]
    degree_in = node_in.bincount(minlength=graph.num_node)
    degree_out = node_out.bincount(minlength=graph.num_node)
    in_order = (node_in * graph.num_node + node_out).argsort()
    out_order = (node_out * graph.num_node + node_in).argsort()
    neighbor_out = node_out[in_order]
    neighbor_in = node_in[out_order]

    # edge_neighbor_out: b that satisfies a -> c and a -> b
    # edge_neighbor_in: b that satisfies a -> c and b -> c
    if edge_ac is not None:
        if edge_ac.dtype == torch.bool:
            edge_ac = edge_ac.nonzero().squeeze(-1)
        node_in = node_in[edge_ac]
        node_out = node_out[edge_ac]
    else:
        edge_ac = torch.arange(graph.num_edge, device=graph.device)
    edge_neighbor_out, edge_degree_in = variadic_index(neighbor_out, degree_in, node_in)
    edge_neighbor_in, edge_degree_out = variadic_index(neighbor_in, degree_out, node_out)
    edge_in_order = variadic_index(in_order, degree_in, node_in)[0]
    edge_out_order = variadic_index(out_order, degree_out, node_out)[0]
    assert (edge_neighbor_in < graph.num_node).all()
    assert (edge_neighbor_out < graph.num_node).all()
    boundary = edge_neighbor_in + graph.num_node * functional._size_to_index(edge_degree_out)
    assert (boundary.diff() >= 0).all()
    key = edge_neighbor_out + graph.num_node * functional._size_to_index(edge_degree_in)
    left = torch.bucketize(key, boundary)
    right = torch.bucketize(key, boundary, right=True)
    count = right - left
    count_per_edge = functional.variadic_sum(count, edge_degree_in)
    sample_per_edge = count_per_edge
    if max_triangle_per_edge is not None:
        sample_per_edge = sample_per_edge.clamp(max=max_triangle_per_edge)
        rand = torch.rand(sample_per_edge.sum(), device=graph.device)
        randint = (rand * count_per_edge.repeat_interleave(sample_per_edge)).long()
    else:
        randint = functional.variadic_arange(count_per_edge)
    offset = count_per_edge.max()
    cum_count_shifted = (count.cumsum(0) - count) - \
                        (count_per_edge.cumsum(0) - count_per_edge).repeat_interleave(edge_degree_in)
    assert (cum_count_shifted <= offset).all()
    assert (randint < offset).all()
    boundary = cum_count_shifted + offset * functional._size_to_index(edge_degree_in)
    key = randint + offset * functional._size_to_index(sample_per_edge)
    index = torch.bucketize(key, boundary, right=True) - 1

    edge_ab = edge_in_order[index]
    # assert (randint >= cum_count_shifted[index]).all()
    # assert (randint < cum_count_shifted[index] + count[index]).all()
    edge_bc = edge_out_order[randint - cum_count_shifted[index] + left[index]]
    edge_ac = edge_ac.repeat_interleave(sample_per_edge)

    # sanity check
    # check_triangles(graph, edge_ab, edge_bc, edge_ac)

    return edge_ab, edge_bc, edge_ac
