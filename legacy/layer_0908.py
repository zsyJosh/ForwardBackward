import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter

from torchdrug import layers, utils
from torchdrug.layers import functional


class LogicMessagePassingConv(layers.MessagePassingBase):

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    aggregate2sum = {
        "sum": "add",
        "mean": "add",
        "max": "max",
    }

    def __init__(self, input_dim, output_dim, num_relation, message_func="distmult", aggregate_func="sum",
                 layer_norm=False, activation="relu", dependent=False, triangle_dropout=0, pre_activation=False,
                 separate_fact_query=False, dependent_add=False, dependent_cat=False, query_fuse=False):
        super(LogicMessagePassingConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.triangle_dropout = triangle_dropout
        self.pre_activation = pre_activation
        self.separate_fact_query = separate_fact_query
        self.dependent_add = dependent_add
        self.dependent_cat = dependent_cat
        self.query_fuse = query_fuse

        if layer_norm:
            if pre_activation:
                self.layer_norm = nn.LayerNorm(input_dim * 2)
            else:
                self.layer_norm = nn.LayerNorm(output_dim)
                if separate_fact_query:
                    self.fact_layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
            self.fact_layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.linear = nn.Linear(input_dim * 2, output_dim)
        if separate_fact_query:
            self.fact_combine_linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.fact_linear = nn.Linear(input_dim, num_relation * input_dim)
        if dependent_cat:
            self.fff_dep_cat = nn.Linear(input_dim * 2, output_dim)
        if self.query_fuse:
            if self.pre_activation:
                self.gamma = nn.Linear(input_dim, num_relation * input_dim * 2)
                self.beta = nn.Linear(input_dim, num_relation * input_dim * 2)
            else:
                self.gamma = nn.Linear(input_dim, num_relation * input_dim)
                self.beta = nn.Linear(input_dim, num_relation * input_dim)
        if self.aggregate_func.startswith("att"):
            self.attention_mlp = layers.MLP(input_dim * 2, [input_dim, 1])

    def message(self, graph, input):
        assert self.triangle_dropout == 0
        # if self.training:
        #     self.mask = torch.rand(len(graph.edge_ac), device=self.device) <= 1 - self.triangle_dropout
        # else:
        #     self.mask = torch.ones(len(graph.edge_ac), dtype=torch.bool, device=self.device)
        if self.dependent:
            fact = self.fact_linear(graph.query).view(-1, self.num_relation, self.input_dim)
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
            if self.dependent_add:
                fact = fact + input
            elif self.dependent_cat:
                fact = self.fff_dep_cat(torch.cat([input, fact], dim=-1))
            input = torch.where(graph.is_query.unsqueeze(-1), input, fact)

        if self.message_func == "transe":
            message = input[graph.edge_ab] + input[graph.edge_bc]
        elif self.message_func == "distmult":
            message = input[graph.edge_ab] * input[graph.edge_bc]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func.startswith("att"):
            query = input[graph.edge_ac]
            key = message
            weight = self.attention_mlp(torch.cat([query, key], dim=-1))
            attention = F.sigmoid(weight)
            message = attention * message

        return message

    def aggregate(self, graph, message):
        edge_weight = (graph.edge_weight[graph.edge_ab] * graph.edge_weight[graph.edge_bc]).unsqueeze(-1)

        aggregate_func = self.aggregate_func.split("-")[-1]
        if aggregate_func in ["sum", "mean", "max"]:
            update = scatter(message * edge_weight, graph.edge_ac, dim=0, dim_size=graph.num_edge,
                             reduce=aggregate_func)
        else:
            raise ValueError("Unknown aggregate function `%s`" % aggregate_func)
        return update

    def message_and_aggregate(self, graph, input):
        if "att" in self.aggregate_func:
            return super(LogicMessagePassingConv, self).message_and_aggregate(graph, input)

        assert self.triangle_dropout == 0

        if self.dependent:
            fact = self.fact_linear(graph.query).view(-1, self.num_relation, self.input_dim)
            relation = graph.edge_list[:, 2]
            sample = graph.edge2graph
            fact = fact[sample, relation]
            if self.dependent_add:
                fact = fact + input
            elif self.dependent_cat: 
                fact = self.fff_dep_cat(torch.cat([input, fact], dim=-1))
            input = torch.where(graph.is_query.unsqueeze(-1), input, fact)

        # PyTorch's coalesce fails when graph.num_edge ** 3 > max torch.long
        # manually coalesce the tensor instead
        key = graph.edge_ac * graph.num_edge + graph.edge_ab
        order = key.argsort()
        indices = torch.stack([graph.edge_ac, graph.edge_ab, graph.edge_bc])
        values = graph.edge_weight[graph.edge_ab] * graph.edge_weight[graph.edge_bc]
        adjacency = utils.sparse_coo_tensor(indices[:, order], values[order],
                                            (graph.num_edge, graph.num_edge, graph.num_edge))
        adjacency._coalesced_(True)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func in self.aggregate2sum:
            sum = self.aggregate2sum[self.aggregate_func]
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        update = functional.generalized_rspmm(adjacency, input, input, sum=sum, mul=mul)
        if self.aggregate_func == "mean":
            degree = graph.edge_ac.bincount(minlength=graph.num_edge)
            update = update / degree

        return update

    def combine(self, graph, input, update):
        if self.pre_activation:
            output = torch.cat([input, update], dim=-1)
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.query_fuse:
                gamma = self.gamma(graph.query).view(-1, self.num_relation, self.input_dim * 2)
                beta = self.beta(graph.query).view(-1, self.num_relation, self.input_dim * 2)
                relation = graph.edge_list[:, 2]
                sample = graph.edge2graph
                gamma = gamma[sample, relation]
                beta = beta[sample, relation]
                output = output * gamma + beta
            if self.activation:
                output = self.activation(output)
            output = self.linear(output)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            # TODO: layer norm is too slow here
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.query_fuse:
                gamma = self.gamma(graph.query).view(-1, self.num_relation, self.input_dim)
                beta = self.beta(graph.query).view(-1, self.num_relation, self.input_dim)
                relation = graph.edge_list[:, 2]
                sample = graph.edge2graph
                gamma = gamma[sample, relation]
                beta = beta[sample, relation]
                output = output * gamma + beta
            if self.activation:
                output = self.activation(output)
            if self.separate_fact_query:
                fact_output = self.fact_combine_linear(torch.cat([input, update], dim=-1))
                if self.fact_layer_norm:
                    fact_output = self.fact_layer_norm(fact_output)
                if self.activation:
                    fact_output = self.activation(fact_output)
                output = torch.where(graph.is_query.unsqueeze(-1), output, fact_output)
        return output

    def forward(self, graph, input):
        update = self.message_and_aggregate(graph, input)
        output = self.combine(graph, input, update)
        return output