import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, utils
from torchdrug.layers import functional


class LogicMessagePassingConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, message_func="distmult", aggregate_func="sum",
                 layer_norm=False, activation="relu", dependent=False, pre_activation=False, message_trans=False):
        super(LogicMessagePassingConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.pre_activation = pre_activation
        self.message_trans = message_trans
        if layer_norm:
            if pre_activation:
                if self.aggregate_func == "pna":
                    self.layer_norm = nn.LayerNorm(input_dim * 13)
                else:
                    self.layer_norm = nn.LayerNorm(input_dim * 2)
            else:
                self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
            self.fact_layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.fact_linear = nn.Linear(input_dim, num_relation * input_dim)
        if self.aggregate_func.startswith("att"):
            self.attention_mlp = layers.MLP(input_dim * 2, [input_dim, 1])
        if self.message_trans:
            self.left_linear = nn.Linear(input_dim, input_dim)
            self.right_linear = nn.Linear(input_dim, input_dim)

    def message(self, graph, input):
        if self.dependent:
            fact = self.fact_linear(graph.query).view(-1, self.num_relation, self.input_dim)
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
            is_fact = ~(graph.is_auxiliary | graph.is_query)
            input = torch.where(is_fact.unsqueeze(-1), fact, input)

        if self.message_func == "transe":
            if self.message_trans:
                message = self.activation(self.left_linear(input[graph.edge_ab])) + self.activation(self.right_linear(input[graph.edge_bc]))
            else:
                message = input[graph.edge_ab] + input[graph.edge_bc]
        elif self.message_func == "distmult":
            if self.message_trans:
                message = self.activation(self.left_linear(input[graph.edge_ab])) * self.activation(self.right_linear(input[graph.edge_bc]))
            else:
                message = input[graph.edge_ab] * input[graph.edge_bc]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func.startswith("att"):
            query = input[graph.edge_ac]
            key = message
            weight = self.attention_mlp(torch.cat([query, key], dim=-1))
            attention = F.sigmoid(weight)
            message = attention * message
        message = torch.cat([message, input])

        return message

    def aggregate(self, graph, message):
        edge_ac = torch.cat([graph.edge_ac, torch.arange(graph.num_edge, device=self.device)])
        edge_weight = graph.edge_weight[graph.edge_ab] * graph.edge_weight[graph.edge_bc]
        edge_weight = torch.cat([edge_weight, torch.ones(graph.num_edge, device=self.device)])
        degree = scatter_add(edge_weight, edge_ac, dim_size=graph.num_edge)
        edge_weight = edge_weight.unsqueeze(-1)
        degree = degree.unsqueeze(-1)

        aggregate_func = self.aggregate_func.split("-")[-1]
        if aggregate_func == "sum":
            update = scatter_add(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)
        elif aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)
        elif aggregate_func == "max":
            update = scatter_max(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)[0]
        elif aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)
            sq_mean = scatter_mean(message ** 2 * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)
            max = scatter_max(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)[0]
            min = scatter_min(message * edge_weight, edge_ac, dim=0, dim_size=graph.num_edge)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if "att" in self.aggregate_func:
            return super(LogicMessagePassingConv, self).message_and_aggregate(graph, input)

        if self.dependent:
            fact = self.fact_linear(graph.query).view(-1, self.num_relation, self.input_dim)
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
            is_fact = ~(graph.is_auxiliary | graph.is_query)
            input = torch.where(is_fact.unsqueeze(-1), fact, input)

        # PyTorch's coalesce fails when graph.num_edge ** 3 > max torch.long
        # manually coalesce the tensor instead
        key = graph.edge_ac * graph.num_edge + graph.edge_ab
        order = key.argsort()
        indices = torch.stack([graph.edge_ac, graph.edge_ab, graph.edge_bc])
        values = graph.edge_weight[graph.edge_ab] * graph.edge_weight[graph.edge_bc]
        adjacency = utils.sparse_coo_tensor(indices[:, order], values[order],
                                            (graph.num_edge, graph.num_edge, graph.num_edge))
        adjacency._coalesced_(True)
        degree = scatter_add(values, graph.edge_ac, dim_size=graph.num_edge)
        degree = degree.unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, input, input, sum="add", mul=mul)
            update = update + input
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, input, input, sum="add", mul=mul)
            update = (update + input) / degree
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, input, input, sum="max", mul=mul)
            update = torch.max(update, input)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, input, input, sum="min", mul=mul)
            mean = (sum + input) / degree
            sq_mean = (sq_sum + input ** 2) / degree
            max = torch.max(max, input)
            min = torch.min(min, input)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree.log()
            if hasattr(graph, "log_degree"):
                scale = scale / graph.log_degree
            else:
                scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def combine(self, input, update):
        if self.pre_activation:
            output = torch.cat([input, update], dim=-1)
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)
            output = self.linear(output)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            # TODO: layer norm is too slow here
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)
        return output


class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", pre_activation=False,
                 relation_transform=False, dropout=0):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.pre_activation = pre_activation
        self.relation_transform = relation_transform
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
                self.layer_norm = nn.LayerNorm(input_dim)
                if self.aggregate_func == "pna":
                    self.layer_norm2 = nn.LayerNorm(input_dim * 12)
                else:
                    self.layer_norm2 = nn.LayerNorm(input_dim)
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
                self.linear = nn.Linear(input_dim * 12, input_dim)
            else:
                self.linear = nn.Linear(input_dim, input_dim)
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
        if self.pre_activation in ["LN", "Transformer"]:
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
            # edge_key = self.key_linear2(torch.cat([edge_input, self.loop_relation(loop)]))
            edge_key = torch.cat([self.relation_key.weight[relation], self.loop_relation.weight[loop]])
            weight = (node_query * edge_key).sum(dim=-1) / (self.input_dim ** 0.5)
            weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
            attention = weight.exp()
            normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
            attention = attention / (normalizer + 1e-10)
            attention = self.dropout(attention)
            message = attention.unsqueeze(-1) * message
        else:
            mask = torch.ones(len(message))
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
        if graph.requires_grad or self.message_func == "rotate" or self.aggregate_func.startswith("att"):
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        boundary = graph.boundary
        if self.relation_transform and hasattr(graph, "relation"):
            relation_input = graph.relation
        else:
            relation_input = self.relation.weight
        if self.relation_transform:
            relation_output = self.relation_layer_norm(relation_input)
            relation_output = self.relation_mlp(relation_output)
            graph.relation = relation_output + relation_input
        if self.pre_activation in ["LN", "Transformer"]:
            input = self.layer_norm(input)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        adjacency = graph.adjacency.transpose(0, 1)

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
            if self.aggregate_func != "pna":
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
