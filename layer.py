from collections import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import data, layers, utils
from torchdrug.layers import functional


class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True, transformer_combine=False,
                 bias=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.transformer_combine = transformer_combine

        feature_dim = input_dim * 12 if aggregate_func == "pna" else input_dim
        if layer_norm:
            if transformer_combine:
                self.layer_norm = nn.LayerNorm(input_dim + feature_dim)
            else:
                self.layer_norm = nn.LayerNorm(output_dim)
            # if not bias:
            #     self.layer_norm.register_parameter("bias", None)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if transformer_combine:
            self.mlp = MultiLayerPerceptron(input_dim + feature_dim, [input_dim, output_dim],
                                            activation=activation, bias=bias)
        else:
            self.linear = nn.Linear(input_dim + feature_dim, output_dim, bias=bias)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim, bias=bias)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
        else:
            relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if isinstance(graph, data.PackedGraph):
            edge_weight = edge_weight.unsqueeze(-1)
            degree_in = graph.degree_in.unsqueeze(-1)
            degree_out = graph.degree_out.unsqueeze(-1)

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "ppr":
            norm = degree_in[node_in]
            norm[-graph.num_node:] = 1
            update = scatter_add(message / norm * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "logsum":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update = update * degree_out.log()
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)
        node_in, node_out, relation = graph.edge_list.t()

        degree_in = graph.degree_in.unsqueeze(-1) + 1
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
            adjacency = utils.sparse_coo_tensor(torch.stack([node_out, node_in, relation]),
                                                graph.edge_weight,
                                                (graph.num_node, graph.num_node, len(graph) * graph.num_relation))
        else:
            relation_input = relation_input.transpose(0, 1).flatten(1)
            adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "ppr":
            input = input / degree_in
            # boundary = boundary / degree_in
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "logsum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out * degree_out.log()
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale_mean = scatter_mean(scale, graph.node2graph, dim=0, dim_size=len(graph))
            scale = scale / scale_mean[graph.node2graph]
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        if isinstance(graph, data.PackedGraph):
            return update
        else:
            return update.view(len(update), batch_size, -1)

    def combine(self, input, update):
        output = torch.cat([input, update], dim=-1)
        if self.transformer_combine:
            output = self.layer_norm(output)
            output = self.mlp(output)
        else:
            output = self.linear(output)
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)
        return output


class RelationAttentionConv(layers.MessagePassingBase):

    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, layer_norm=False, activation="relu",
                 dependent=True, dropout=0, input_as_boundary=False, transform_node_value=False, no_boundary=False,
                 num_head=1, loop_dependent=False, aggregation="mean", no_attention=False, single_layer=False,
                 combine_gate=False, positional_embedding=False):
        super(RelationAttentionConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.dependent = dependent
        self.input_as_boundary = input_as_boundary
        self.transform_node_value = transform_node_value
        self.no_boundary = no_boundary
        self.num_head = num_head
        self.loop_dependent = loop_dependent
        self.aggregation = aggregation
        self.no_attention = no_attention
        self.single_layer = single_layer
        self.combine_gate = combine_gate
        assert aggregation in ["sum", "mean"]
        assert not (input_as_boundary and no_boundary)

        self.mp_layer_norm = None
        self.mlp_layer_norm = None
        if layer_norm == True:
            self.mp_layer_norm = nn.LayerNorm(input_dim)
            self.mlp_layer_norm = nn.LayerNorm(input_dim * (1 if combine_gate else 2))
        elif layer_norm == "mp":
            self.mp_layer_norm = nn.LayerNorm(input_dim)
        elif layer_norm == "mlp":
            self.mlp_layer_norm = nn.LayerNorm(input_dim * (1 if combine_gate else 2))
        if single_layer:
            self.linear = nn.Linear(input_dim * (1 if combine_gate else 2), input_dim)
        else:
            self.mlp = layers.MLP(input_dim * (1 if combine_gate else 2), [input_dim, output_dim], activation=activation)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        if dependent:
            self.relation_key_linear = nn.Linear(query_input_dim, num_relation * input_dim)
            self.relation_value_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation_key = nn.Embedding(num_relation, input_dim)
            self.relation_value = nn.Embedding(num_relation, input_dim)
        if loop_dependent:
            self.loop_key_linear = nn.Linear(query_input_dim, input_dim)
            self.loop_value_linear = nn.Linear(query_input_dim, input_dim)
        else:
            self.loop_key = nn.Embedding(1, input_dim)
            self.loop_value = nn.Embedding(1, input_dim)
        if combine_gate:
            self.gate_linear = nn.Linear(input_dim * 2, input_dim)
        if positional_embedding:
            self.positional_embedding = PositionalEmbedding(input_dim)
        else:
            self.positional_embedding = None

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)

        query = graph.query
        if self.positional_embedding:
            position = torch.ones(*query.shape[:-1], device=self.device) * graph.iteration_id
            position_input = self.positional_embedding(position)
            query = query + position_input

        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_value = self.relation_value_linear(query).view(batch_size, self.num_relation, self.input_dim)
            relation_key = self.relation_key_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_value = self.relation_value.weight.expand(batch_size, -1, -1)
            relation_key = self.relation_key.weight.expand(batch_size, -1, -1)
        if self.loop_dependent:
            loop_value = self.loop_value_linear(query)
            loop_key = self.loop_key_linear(query)
        else:
            loop_value = self.loop_value.weight.expand(batch_size, -1)
            loop_key = self.loop_key.weight.expand(batch_size, -1)
        relation_value = relation_value.transpose(0, 1)
        relation_key = relation_key.transpose(0, 1)
        if self.mp_layer_norm:
            input = self.mp_layer_norm(input)

        # if self.positional_embedding:
        #     position = torch.ones(*relation_value.shape[:-1], device=self.device) * graph.iteration_id
        #     position_input = self.positional_embedding(position)
        #     relation_value = relation_value + position_input

        # check if value_linear layer is necessary
        node_key = self.key_linear(input)[node_in]
        if self.transform_node_value:
            node_value = self.value_linear(input)[node_in]
        else:
            node_value = input[node_in]

        edge_value = relation_value[relation]
        value = node_value * edge_value

        # check if boundary condition is necessary
        if self.input_as_boundary:
            node_key = torch.cat([node_key, self.key_linear(input)])
            if self.transform_node_value:
                value = torch.cat([value, self.value_linear(input) * loop_value])
            else:
                value = torch.cat([value, input * loop_value])
        elif not self.no_boundary:
            node_key = torch.cat([node_key, self.key_linear(graph.boundary)])
            if self.transform_node_value:
                value = torch.cat([value, self.value_linear(graph.boundary)])
            else:
                value = torch.cat([value, graph.boundary])

        if self.no_boundary:
            edge_key = relation_key[relation]
        else:
            node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
            edge_key = torch.cat([relation_key[relation], loop_key.expand(graph.num_node, -1, -1)])
        weight = (node_key * edge_key).view(*node_key.shape[:-1], self.num_head, -1)
        weight = weight.sum(dim=-1) / ((self.input_dim / self.num_head) ** 0.5)
        if self.aggregation == "mean":
            if self.no_attention:
                attention = torch.ones_like(weight)
            else:
                weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
                attention = weight.exp()
            normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
            attention = attention / (normalizer + self.eps)
        elif self.aggregation == "sum":
            if self.no_attention:
                attention = torch.ones_like(weight)
            else:
                attention = F.sigmoid(weight)
        else:
            raise ValueError
        if self.dropout:
            attention = self.dropout(attention)
        attention = attention.unsqueeze(-1).expand(-1, -1, -1, self.input_dim // self.num_head).flatten(-2)
        message = attention * value

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        if not self.no_boundary:
            node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        if self.aggregation == "mean":
            update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregation == "sum":
            update = scatter_add(message, node_out, dim=0, dim_size=graph.num_node)
        else:
            raise ValueError

        # if self.positional_embedding:
        #     position = torch.ones(*update.shape[:-1], device=self.device) * graph.iteration_id
        #     position_input = self.positional_embedding(position)
        #     update = update + position_input

        return update

    def combine(self, input, update):
        # output = input + update
        output = torch.cat([input, update], dim=-1)
        if self.combine_gate:
            prob = F.sigmoid(self.gate_linear(output))
            output = prob * update + (1 - prob) * input
        if self.mlp_layer_norm:
            output = self.mlp_layer_norm(output)
        if self.single_layer:
            output = self.activation(output)
            output = self.linear(output)
        else:
            output = self.mlp(output)
        return output

    def combine_(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.mlp_layer_norm:
            output = self.mlp_layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GeneralizedRelationalConvRelPred(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", pre_activation=False,
                 relation_transform=False, dropout=0, concat_update=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.pre_activation = pre_activation
        self.relation_transform = relation_transform
        self.concat_update = concat_update
        assert pre_activation in ["LN", "Transformer", True, False]

        if layer_norm:
            if self.pre_activation == True:
                if self.aggregate_func == "pna":
                    self.layer_norm = nn.LayerNorm(input_dim * 13)
                else:
                    self.layer_norm = nn.LayerNorm(input_dim * 2)
            elif self.pre_activation == "LN":
                self.layer_norm = nn.LayerNorm(input_dim)
            elif self.pre_activation == "Transformer":
                if layer_norm == "mlp":
                    self.layer_norm = None
                else:
                    self.layer_norm = nn.LayerNorm(input_dim)
                if self.aggregate_func == "pna":
                    self.layer_norm2 = nn.LayerNorm(input_dim * (12 + int(concat_update)))
                else:
                    self.layer_norm2 = nn.LayerNorm(input_dim * (1 + int(concat_update)))
            else:
                self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.dropout = nn.Dropout(dropout)

        if self.aggregate_func == "pna" and self.pre_activation != "Transformer":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        elif self.pre_activation == "Transformer":
            if self.aggregate_func == "pna":
                self.linear = nn.Linear(input_dim * (12 + int(concat_update)), input_dim)
            else:
                self.linear = nn.Linear(input_dim * (1 + int(concat_update)), input_dim)
            self.linear2 = nn.Linear(input_dim, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if self.aggregate_func == "attention":
            self.key_linear = nn.Linear(input_dim, input_dim)
            # self.key_linear2 = nn.Linear(input_dim, input_dim)
            self.relation_key = nn.Embedding(num_relation, input_dim)
            self.loop_relation = nn.Embedding(1, input_dim)
        elif self.aggregate_func.startswith("att-"):
            self.attention_mlp = layers.MLP(input_dim * 2, [input_dim, 1])
        self.relation = nn.Embedding(num_relation, input_dim)
        if self.relation_transform:
            self.relation_layer_norm = nn.LayerNorm(input_dim)
            self.relation_mlp = layers.MLP(input_dim, [input_dim, output_dim])

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        if self.relation_transform and hasattr(graph, "relation"):
            relation_input = graph.relation
        else:
            relation_input = self.relation.weight
        if self.relation_transform:
            relation_output = self.relation_layer_norm(relation_input)
            relation_output = self.relation_mlp(relation_output)
            graph.relation = relation_output + relation_input
        if self.pre_activation in ["LN", "Transformer"] and self.layer_norm is not None:
            input = self.layer_norm(input)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        if self.aggregate_func.startswith("att-"):
            query = input[node_out]
            key = message
            weight = self.attention_mlp(torch.cat([query, key], dim=-1))
            attention = F.sigmoid(weight)
            message = attention * message

        message = torch.cat([message, graph.boundary])

        if self.aggregate_func == "attention":
            loop = torch.zeros(graph.num_node, dtype=torch.long, device=self.device)
            node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
            node_query = self.key_linear(torch.cat([node_input, graph.boundary]))
            # edge_key = self.key_linear2(torch.cat([edge_input, self.loop_relation.weight[loop]]))
            # independent relation key
            edge_key = torch.cat([self.relation_key.weight[relation], self.loop_relation.weight[loop]])
            weight = (node_query * edge_key).sum(dim=-1) / (self.input_dim ** 0.5)
            weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
            attention = weight.exp()
            normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
            attention = attention / (normalizer + 1e-10)
            attention = self.dropout(attention)
            message = attention.unsqueeze(-1) * message
        else:
            mask = torch.ones(len(message), device=self.device)
            mask = self.dropout(mask)
            message = mask.unsqueeze(-1) * message

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1

        aggregate_func = self.aggregate_func.split("-")[-1]
        if aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif aggregate_func in ["mean", "attention"]:
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)
        # if graph.requires_grad or self.message_func == "rotate" or self.aggregate_func.startswith("att"):
        #     return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        mask = torch.ones(graph.num_node, 1, device=self.device)
        mask = self.dropout(mask)
        boundary = mask * graph.boundary
        if self.relation_transform and hasattr(graph, "relation"):
            relation_input = graph.relation
        else:
            relation_input = self.relation.weight
        if self.relation_transform:
            relation_output = self.relation_layer_norm(relation_input)
            relation_output = self.relation_mlp(relation_output)
            graph.relation = relation_output + relation_input
        if self.pre_activation in ["LN", "Transformer"] and self.layer_norm is not None:
            input = self.layer_norm(input)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t(), self.dropout(graph.edge_weight), graph.shape)
        adjacency = adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def combine(self, input, update):
        if self.pre_activation == "Transformer":
            if self.concat_update:
                update = torch.cat([input, update], dim=-1)
            elif self.aggregate_func != "pna":
                update = update + input
            output = self.layer_norm2(update)
            output = self.linear(output)
            if self.activation:
                output = self.activation(output)
            output = self.linear2(output)
        elif self.pre_activation:
            # the one used in ResNet v2
            output = torch.cat([input, update], dim=-1)
            if self.pre_activation == True and self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)
            output = self.linear(output)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)

        return output


class PositionalEmbedding(nn.Module):

    def __init__(self, output_dim):
        super(PositionalEmbedding, self).__init__()
        inv_frequency = 1 / (10000 ** (torch.arange(0, output_dim // 2) / (output_dim // 2 - 1)))
        self.register_buffer("inv_frequency", inv_frequency)

    def forward(self, position):
        time = position.unsqueeze(-1) * self.inv_frequency
        signal = torch.cat([time.sin(), time.cos()], dim=-1)
        return signal


class MultiLayerPerceptron(layers.MLP):

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0,
                 bias=True):
        nn.Module.__init__(self)

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1], bias=bias))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None