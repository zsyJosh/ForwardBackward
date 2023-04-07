import logging
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, segment_add_coo

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.utils import comm
from torchdrug.core import Registry as R

import layer

logger = logging.getLogger(__name__)


@R.register("model.RuleIdentity")
class RuleIdentity(nn.Module, core.Configurable):

    def __init__(self, input_dim, num_relation):
        super(RuleIdentity, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation

        self.relation = nn.Embedding(num_relation * 2, input_dim)

    def embed(self, r_index):
        return self.relation(r_index)

    def forward(self, query, all_loss=None, metric=None):
        subgoals = query.unsqueeze(-2)
        masks = torch.ones(subgoals.shape[:-1], dtype=torch.bool, device=self.device)
        return subgoals, masks


@R.register("model.RuleRNN")
class RuleRNN(nn.Module, core.Configurable):

    def __init__(self, input_dim, num_relation, num_hop, straight_through=False, gumbel_softmax=False, tau=1,
                 vector_quantization=False, vq_loss_weight=1, beta=0.25, autoregressive=True, identity=False,
                 num_gru_layer=1):
        super(RuleRNN, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.num_hop = num_hop
        self.straight_through = straight_through
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau
        self.vector_quantization = vector_quantization
        self.vq_loss_weight = vq_loss_weight
        self.beta = beta
        self.autoregressive = autoregressive
        self.identity = identity

        self.gru = nn.GRU(input_dim, input_dim, num_gru_layer)
        if identity:
            self.mlp = layers.MLP(input_dim, [input_dim, 1])
        self.relation = nn.Embedding(num_relation * 2, input_dim)
        self.eos = nn.Embedding(1, input_dim)

    def embed(self, r_index):
        return self.relation(r_index)

    def autoregressive_forward(self, query, all_loss=None, metric=None):
        shape = query.shape
        query = query.flatten(0, -2)
        eos = self.eos.weight.expand_as(query)
        input = torch.stack([query, eos], dim=0)
        hidden, hx = self.gru(input)
        hidden = hidden[1:]
        subgoals = []

        for i in range(self.num_hop):
            hidden = hidden.squeeze(0)
            if self.vector_quantization:
                score = -(hidden.unsqueeze(-2) - self.relation.weight).norm(dim=-1)
            else:
                score = torch.einsum("bd, rd -> br", hidden, self.relation.weight)
            if self.gumbel_softmax:
                attention = F.gumbel_softmax(score, tau=self.tau, dim=-1)
            elif self.vector_quantization:
                attention = torch.zeros_like(score)
                attention.scatter_(1, score.argmax(dim=-1, keepdim=True), 1)
            else:
                attention = F.softmax(score / self.tau, dim=-1)
            if self.straight_through:
                sample = torch.multinomial(attention, 1)
                one = torch.ones(sample.shape, device=sample.device)
                sample = scatter_add(one, sample, dim=-1, dim_size=self.num_relation * 2)
                attention = (sample - attention).detach() + attention
            subgoal = torch.einsum("br, rd -> bd", attention, self.relation.weight)

            if self.vector_quantization and all_loss is not None:
                embedding_loss = F.mse_loss(subgoal, hidden.detach())
                commitment_loss = F.mse_loss(subgoal.detach(), hidden)
                metric["VQ loss"] = embedding_loss
                loss = commitment_loss * self.beta + embedding_loss
                all_loss += loss * self.vq_loss_weight
                # straight through estimator
                subgoal = (subgoal - hidden).detach() + hidden

            subgoals.append(subgoal)
            hidden, hx = self.gru(subgoal.unsqueeze(0), hx)
        subgoals = torch.stack(subgoals, dim=-2)

        if self.identity:
            prob = F.sigmoid(self.mlp(query)).unsqueeze(-2)
            identity = query.unsqueeze(-2).expand_as(subgoals)
            if self.straight_through:
                sample = (torch.rand_like(prob) < prob).float()
                prob = (sample - prob).detach() + prob
            subgoals = prob * identity + (1 - prob) * subgoals
        subgoals = subgoals.view(*shape[:-1], self.num_hop, shape[-1])

        return subgoals

    def oneshot_forward(self, query, all_loss=None, metric=None):
        # one-shot forward doesn't support gumbel softmax
        assert not self.gumbel_softmax and not self.straight_through

        shape = query.shape
        query = query.flatten(0, -2)
        eos = self.eos.weight.expand_as(query)
        input = torch.stack([query] * (self.num_hop - 1) + [eos], dim=0)
        subgoals = self.gru(input)[0]

        if self.identity:
            prob = F.sigmoid(self.mlp(query)).unsqueeze(-2)
            identity = query.unsqueeze(-2).expand_as(subgoals)
            if self.straight_through:
                sample = (torch.rand_like(prob) < prob).float()
                prob = (sample - prob).detach() + prob
            subgoals = prob * identity + (1 - prob) * subgoals
        subgoals = subgoals.view(*shape[:-1], self.num_hop, shape[-1])

        return subgoals.transpose(0, -2)

    def forward(self, query, all_loss=None, metric=None):
        if self.autoregressive:
            subgoals = self.autoregressive_forward(query, all_loss, metric)
        else:
            subgoals = self.oneshot_forward(query, all_loss, metric)

        masks = torch.ones(subgoals.shape[:-1], dtype=torch.bool, device=self.device)
        return subgoals, masks


@R.register("model.RuleEmbedding")
class RuleEmbedding(nn.Module, core.Configurable):

    def __init__(self, input_dim, num_relation, num_hop, message_func="distmult", straight_through=False,
                 gumbel_softmax=False, tau=1, num_mlp_layer=2):
        super(RuleEmbedding, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.num_hop = num_hop
        self.message_func = message_func
        self.straight_through = straight_through
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau

        self.relation = nn.Embedding(num_relation * 2, input_dim)
        self.mlp = layers.MLP(input_dim * 2, [input_dim] * (num_mlp_layer - 1) + [1])

    def embed(self, r_index):
        return self.relation(r_index)

    def forward(self, query, all_loss=None, metric=None):
        r_index = torch.arange(self.num_relation * 2, device=self.device)
        r_indexes = [r_index] * self.num_hop
        r_indexes = torch.meshgrid(*r_indexes)
        r_indexes = torch.stack(r_indexes, dim=-1)
        r = self.relation(r_indexes)  # (R, ..., R, num_hop, d)
        if self.message_func == "transe":
            body = r.sum(dim=-2)  # (R, ..., R, d)
        elif self.message_func == "distmult":
            body = r.prod(dim=-2)

        shape = list(query.shape[:-1]) + [1] * self.num_hop + [query.shape[-1]]
        repeat = [-1] * (query.ndim - 1) + [self.num_relation * 2] * self.num_hop + [-1]
        query = query.view(shape)
        query = query.expand(repeat)  # (batch_size, R, ..., R, d)
        body = body.expand_as(query)  # (batch_size, R, ..., R, d)
        feature = torch.cat([query, body], dim=-1)
        score = self.mlp(feature).squeeze(-1)  # (batch_size, R, ..., R)

        score = score.flatten(-self.num_hop) # (batch_size, R ** num_hop)
        num_rule = score.shape[-1]
        r = r.flatten(0, -3)  # (R ** num_hop, num_hop, d)
        if self.gumbel_softmax:
            attention = F.gumbel_softmax(score, tau=self.tau, dim=-1)
        else:
            attention = F.softmax(score / self.tau, dim=-1)
        if self.straight_through:
            sample = torch.multinomial(attention, 1)
            one = torch.ones(sample.shape, device=sample.device)
            sample = scatter_add(one, sample, dim=-1, dim_size=num_rule)
            attention = (sample - attention).detach() + attention
        subgoals = (attention.unsqueeze(-1).unsqueeze(-1) * r).sum(dim=-3)
        masks = torch.ones(subgoals.shape[:-1], dtype=torch.bool, device=self.device)
        return subgoals, masks


@R.register("model.RuleNBFNet")
class RuleNBFNet(nn.Module, core.Configurable):

    def __init__(self, input_dim, num_relation, num_hop, message_func="distmult", aggregate_func="pna", short_cut=False,
                 layer_norm=False, activation="relu", dependent=True, transformer_combine=False, straight_through=False,
                 gumbel_softmax=False, tau=1, num_mlp_layer=2, query_as_indicator=False):
        super(RuleNBFNet, self).__init__()
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.num_hop = num_hop
        self.short_cut = short_cut
        self.straight_through = straight_through
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau
        self.num_mlp_layer = num_mlp_layer
        self.query_as_indicator = query_as_indicator

        self.layers = nn.ModuleList()
        for i in range(num_hop):
            self.layers.append(layer.GeneralizedRelationalConv(input_dim, input_dim, num_relation * 2, input_dim,
                                                               message_func, aggregate_func, layer_norm,
                                                               activation, dependent, transformer_combine))
        self.relation = nn.Embedding(num_relation * 2, input_dim)
        self.indicator = nn.Embedding(1, input_dim)
        self.mlp = layers.MLP(input_dim * 2, [input_dim] * (num_mlp_layer - 1) + [1])

    def embed(self, r_index):
        return self.relation(r_index)

    def bellmanford(self, graph, h_index, query, t_index):
        index = h_index + graph.num_cum_nodes - graph.num_nodes
        boundary = torch.zeros(graph.num_node, self.input_dim, device=self.device)
        if self.query_as_indicator:
            boundary[index] = query
        else:
            boundary[index] = self.indicator.weight
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = query[graph.node2graph]
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        index = t_index + graph.num_cum_nodes - graph.num_nodes
        return output[index]

    def forward(self, query, all_loss=None, metric=None):
        r_index = torch.arange(self.num_relation * 2, device=self.device)
        r_indexes = [r_index] * self.num_hop
        r_indexes = torch.meshgrid(*r_indexes)
        r_indexes = torch.stack(r_indexes, dim=-1)
        r = self.relation(r_indexes)
        r = r.flatten(0, -3)
        num_rule = len(r)

        node_in = torch.arange(self.num_hop, device=self.device)
        node_out = torch.arange(self.num_hop, device=self.device) + 1
        node_in = node_in.repeat(num_rule)
        node_out = node_out.repeat(num_rule)
        relation = r_indexes.flatten()
        edge_list = torch.stack([node_in, node_out, relation], dim=-1)
        num_nodes = torch.ones(num_rule, dtype=torch.long, device=self.device) * (self.num_hop + 1)
        num_edges = torch.ones(num_rule, dtype=torch.long, device=self.device) * self.num_hop
        # TODO: do we need to apply undirected()?
        graph = data.PackedGraph(edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                 num_relation=self.num_relation * 2)

        shape = list(query.shape[:-1]) + [num_rule]
        repeat = [-1] * (query.ndim - 1) + [num_rule, -1]
        graph = graph.repeat(len(query))
        query = query.unsqueeze(-2).expand(repeat)
        query = query.flatten(0, -2)
        h_index = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        t_index = torch.ones(len(graph), dtype=torch.long, device=self.device) * self.num_hop
        feature = self.bellmanford(graph, h_index, query, t_index)
        score = self.mlp(feature).squeeze(-1)
        score = score.view(shape)

        if self.gumbel_softmax:
            attention = F.gumbel_softmax(score, tau=self.tau, dim=-1)
        else:
            attention = F.softmax(score / self.tau, dim=-1)
        if self.straight_through:
            sample = torch.multinomial(attention, 1)
            one = torch.ones(sample.shape, device=sample.device)
            sample = scatter_add(one, sample, dim=-1, dim_size=num_rule)
            attention = (sample - attention).detach() + attention
        subgoals = (attention.unsqueeze(-1).unsqueeze(-1) * r).sum(dim=-3)
        masks = torch.ones(subgoals.shape[:-1], dtype=torch.bool, device=self.device)
        return subgoals, masks


@R.register("model.RuleFromModel")
class RuleFromModel(nn.Module, core.Configurable):

    def __init__(self, input_dim, num_relation, num_hop, straight_through=False, gumbel_softmax=False, tau=1):
        super(RuleFromModel, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.num_hop = num_hop
        self.straight_through = straight_through
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau

        self.relation = nn.Embedding(num_relation * 2, input_dim)
        range = torch.arange(num_relation * 2)
        value = torch.ones(num_relation * 2) * 1e9
        # identity prior
        score = utils.sparse_coo_tensor(torch.stack([range] * (num_hop + 1)), value, [num_relation * 2] * (num_hop + 1))
        score = score.to_dense()
        self.register_buffer("score", score)

    def embed(self, r_index):
        return self.relation(r_index)

    def forward(self, query, all_loss=None, metric=None):
        r_index = torch.arange(self.num_relation * 2, device=self.device)
        r_indexes = [r_index] * self.num_hop
        r_indexes = torch.meshgrid(*r_indexes)
        r_indexes = torch.stack(r_indexes, dim=-1)
        r = self.relation(r_indexes)  # (R, ..., R, num_hop, d)
        r = r.flatten(0, -3)  # (R ** num_hop, num_hop, d)
        num_rule = len(r)

        # nearest neighbor
        r_index = (query.unsqueeze(-2) - self.relation.weight).norm(dim=-1).argmin(dim=-1) # (batch_size,)
        score = self.score[r_index]
        score = score.flatten(1)
        if self.gumbel_softmax:
            attention = F.gumbel_softmax(score, tau=self.tau, dim=-1)
        else:
            attention = F.softmax(score / self.tau, dim=-1)
        if self.straight_through:
            sample = torch.multinomial(attention, 1)
            one = torch.ones(sample.shape, device=sample.device)
            sample = scatter_add(one, sample, dim=-1, dim_size=num_rule)
            attention = (sample - attention).detach() + attention
        subgoals = (attention.unsqueeze(-1).unsqueeze(-1) * r).sum(dim=-3)
        masks = torch.ones(subgoals.shape[:-1], dtype=torch.bool, device=self.device)
        return subgoals, masks

    @torch.no_grad()
    def update_score(self, forward_model, mlp):
        if comm.get_rank() == 0:
            logger.warning("Compute rule score from forward model")
        r_index = torch.arange(self.num_relation * 2, device=self.device)
        r_indexes = [r_index] * self.num_hop
        r_indexes = torch.meshgrid(*r_indexes)
        r_indexes = torch.stack(r_indexes, dim=-1)
        num_rule = len(r_indexes.flatten(0, -2))  # (R ** num_hop, num_hop)

        node_in = torch.arange(self.num_hop, device=self.device)
        node_out = torch.arange(self.num_hop, device=self.device) + 1
        node_in = node_in.repeat(num_rule)
        node_out = node_out.repeat(num_rule)
        relation = r_indexes.flatten()
        inv_relation = (relation + self.num_relation) % (self.num_relation * 2)
        edge_list = torch.stack([node_in, node_out, relation], dim=-1)
        inv_edge_list = torch.stack([node_out, node_in, inv_relation], dim=-1)
        edge_list = torch.stack([edge_list, inv_edge_list], dim=1).flatten(0, 1)
        num_nodes = torch.ones(num_rule, dtype=torch.long, device=self.device) * (self.num_hop + 1)
        num_edges = torch.ones(num_rule, dtype=torch.long, device=self.device) * self.num_hop * 2
        graph = data.PackedGraph(edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                 num_relation=self.num_relation * 2)

        h_index = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        t_index = torch.ones(len(graph), dtype=torch.long, device=self.device) * self.num_hop
        score = []
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        work_load = self.num_relation * 2 // world_size
        if self.num_relation * 2 % world_size > 0:
            work_load += 1
        for i in range(work_load * rank, min(work_load * (rank + 1), self.num_relation * 2)):
            r_index = torch.ones(num_rule, dtype=torch.long, device=self.device) * i
            query = self.relation(r_index)
            with graph.graph():
                graph.query = query

            shard_size = 100000
            num_shard = len(graph) // shard_size
            if len(graph) % shard_size > 0:
                num_shard += 1
            hidden = []
            for start in range(num_shard):
                end = min(start + shard_size, len(graph))
                g = graph[start: end]
                h = forward_model.embed(g, h_index[start: end])
                h = forward_model(g, h)
                hidden.append(h)
            hidden = torch.cat(hidden)
            feature = forward_model.readout(graph, hidden, t_index.unsqueeze(-1)).squeeze(-2)
            feature = torch.cat([feature, query], dim=-1)
            score.append(mlp(feature).squeeze(-1))

        score = torch.stack(score)
        self.score[:] = comm.cat(score).view(self.score.shape)

        if comm.get_rank() == 0:
            logger.warning("Done!")


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, dependent=True,
                 transformer_combine=False):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation * 2,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent, transformer_combine))
        self.indicator = nn.Embedding(1, input_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        if not isinstance(graph, data.PackedGraph):
            input = input.transpose(0, 1)
        with graph.node():
            graph.boundary = input
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if not isinstance(graph, data.PackedGraph):
            node_feature = node_feature.transpose(0, 1)
        return node_feature

    def embed(self, graph, h_index, query=None):
        if query is None:
            indicator = self.indicator(torch.zeros_like(h_index))
        else:
            indicator = query
        if isinstance(graph, data.PackedGraph):
            index = h_index + graph.num_cum_nodes - graph.num_nodes
            input = torch.zeros(graph.num_node, self.input_dim, device=self.device)
            input[index] = indicator
        else:
            index = h_index.unsqueeze(-1).expand_as(indicator)
            input = torch.zeros(graph.num_node, *indicator.shape, device=self.device)
            input.scatter_(0, index.unsqueeze(0), indicator.unsqueeze(0))
            input = input.transpose(0, 1)
        return input

    def readout(self, graph, hidden, t_index):
        if isinstance(graph, data.PackedGraph):
            index = t_index + (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)
            feature = hidden[index]
        else:
            index = t_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            feature = hidden.gather(1, index)
        return feature


@R.register("model.AttentionNBFNet")
class AttentionNBFNet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, short_cut=False, layer_norm=False, activation="relu",
                 tied_weights=False, concat_hidden=False, num_mlp_layer=2, dropout=0, dependent=True,
                 remove_one_hop=False, input_as_boundary=False, transform_node_value=False, no_boundary=False,
                 num_head=1, loop_dependent=False, mlp_dropout=0, aggregation="mean", no_attention=False,
                 single_layer=False, per_step_gate=False, combine_gate=False, positional_embedding=False):
        super(AttentionNBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.per_step_gate = per_step_gate
        if per_step_gate:
            assert tied_weights

        self.layers = nn.ModuleList()
        if tied_weights:
            self.layers += [layer.RelationAttentionConv(self.dims[0], self.dims[1], num_relation * 2, self.dims[0],
                                                        layer_norm, activation, dependent, dropout, input_as_boundary,
                                                        transform_node_value, no_boundary, num_head, loop_dependent,
                                                        aggregation, no_attention, single_layer, combine_gate,
                                                        positional_embedding)] \
                           * len(hidden_dims)
        else:
            for i in range(len(hidden_dims)):
                self.layers.append(layer.RelationAttentionConv(self.dims[i], self.dims[i + 1], num_relation * 2,
                                                               self.dims[0], layer_norm, activation, dependent,
                                                               dropout, input_as_boundary, transform_node_value,
                                                               no_boundary, num_head, loop_dependent, aggregation,
                                                               no_attention, single_layer, combine_gate,
                                                               positional_embedding))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(num_relation * 2, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1], dropout=mlp_dropout)
        if per_step_gate:
            self.gate = layers.MLP(input_dim * 2, [input_dim * 2] * (num_mlp_layer - 1) + [1])

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

    @utils.cached
    def bellmanford(self, graph, h_index, r_index):
        batch_size = len(h_index)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
        node_query = query.expand(graph.num_node, -1, -1)

        hiddens = []
        layer_input = boundary
        sum_hidden = torch.zeros_like(layer_input)
        remainder = torch.ones(graph.num_node, batch_size, 1, device=self.device)

        for i, layer in enumerate(self.layers):
            graph.iteration_id = i
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.per_step_gate:
                x = torch.cat([hidden, node_query], dim=-1)
                prob = F.hardsigmoid(self.gate(x))
                if i == len(self.layers) - 1:
                    prob = torch.ones_like(prob)
                sum_hidden = sum_hidden + remainder * prob * hidden
                remainder = (1 - prob) * remainder
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            if self.per_step_gate:
                output = torch.cat([sum_hidden, node_query], dim=-1)
            else:
                output = torch.cat([hiddens[-1], node_query], dim=-1)

        return output

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        feature = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = feature.transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)

        return score


@R.register("model.NBFNetRelPred")
class NeuralBellmanFordNetworkRelationPrediction(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 tied_weights=False, concat_hidden=False, pre_activation=False, flip_graph=True,
                 relation_transform=False, num_test_layer=None, attend_relation=False, num_latent_relation=0,
                 dropout=0, concat_update=False):
        super(NeuralBellmanFordNetworkRelationPrediction, self).__init__()

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
        self.concat_update = concat_update
        if num_test_layer is not None:
            assert tied_weights

        if flip_graph:
            num_relation *= 2

        self.layers = nn.ModuleList()
        self.query_linears = nn.ModuleList()
        self.relation_keys = nn.ModuleList()
        self.latent_relations = nn.ModuleList()
        if tied_weights:
            self.layers += [layer.GeneralizedRelationalConvRelPred(self.dims[0], self.dims[1], num_relation,
                                                            self.dims[0], message_func, aggregate_func, layer_norm,
                                                            activation, pre_activation, relation_transform, dropout,
                                                            concat_update)] \
                           * len(hidden_dims)
            if attend_relation:
                self.query_linears += [nn.Linear(self.dims[0], self.dims[0])] * len(hidden_dims)
                self.relation_keys += [nn.Embedding(num_relation + num_latent_relation, self.dims[0])] * len(hidden_dims)
                self.latent_relations += [nn.Embedding(num_latent_relation, self.dims[0])] * len(hidden_dims)
        else:
            for i in range(len(hidden_dims)):
                self.layers.append(layer.GeneralizedRelationalConvRelPred(self.dims[i], self.dims[i + 1], num_relation,
                                                                   self.dims[0], message_func, aggregate_func,
                                                                   layer_norm, activation, pre_activation,
                                                                   relation_transform, dropout, concat_update))
                if attend_relation:
                    self.query_linears.append(nn.Linear(self.dims[i], self.dims[i]))
                    self.relation_keys.append(nn.Embedding(num_relation + num_latent_relation, self.dims[i]))
                    self.latent_relations.append(nn.Embedding(num_latent_relation, self.dims[i])) * len(hidden_dims)

        self.query = nn.Embedding(1, input_dim)

    def bellmanford(self, graph, h_index):
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

        feature = self.bellmanford(graph, h_index)
        t_index = t_index + graph.num_cum_nodes - graph.num_nodes
        feature = feature[t_index]

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