import os
import csv
import argparse
import logging

import torch
from torchdrug import core, data, metrics, datasets

import dataset
import layer
import model
import framework
import task
import engine
import util

util.setup_debug_hook()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/knowledge_graph/wn18rr.yaml")

    return parser.parse_known_args()[0]


def load_vocab(dataset):
    entity_vocab = dataset.entity_vocab
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]

    return entity_vocab, relation_vocab


def build_solver(cfg):
    if "KnowledgeGraphCompletion" in cfg.task["class"] or "InductiveRelationPrediction" in cfg.task["class"]:
        cfg.task.model.num_relation = _dataset.num_relation
        if "BackwardForwardReasoning" in cfg.task.model["class"]:
            cfg.task.model.backward_model.num_relation = _dataset.num_relation
            cfg.task.model.forward_model.num_relation = _dataset.num_relation
    _task = core.Configurable.load_config_dict(cfg.task)
    _task.model.max_epoch = cfg.train.num_epoch
    _task.model.adversarial_temperature = _task.adversarial_temperature
    cfg.optimizer.params = _task.parameters()
    logger.warning(f"#trainable parameters: {sum(param.numel() for param in _task.parameters())}")
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if "scheduler" in cfg:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
    else:
        scheduler = None
    return engine.EngineEx(_task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config)[0]

    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = _dataset.split()
    entity_vocab, relation_vocab = load_vocab(_dataset)
    relation_vocab = relation_vocab + ["%s^{-1}" % r for r in relation_vocab]

    solver = build_solver(cfg)
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)

    # train graph
    graph = _dataset.graph.edge_mask(slice(0, _dataset.num_samples[0]))
    graph = graph.cuda(solver.device)
    graph = graph.undirected(add_inverse=True)
    rel_ab, rel_bc, rel_ac, coverage, confidence = model.rule_mining(graph, min_coverage=0, min_confidence=0)
    num_rule = len(rel_ab)
    node_in = torch.arange(2, device=solver.device)
    node_out = torch.arange(1, 3, device=solver.device)
    node_in = node_in.repeat(num_rule)
    node_out = node_out.repeat(num_rule)
    relation = torch.stack([rel_ab, rel_bc], dim=-1).flatten()
    edge_list = torch.stack([node_in, node_out, relation], dim=-1)
    inv_relation = (relation + _dataset.num_relation) % (_dataset.num_relation * 2)
    inv_edge_list = torch.stack([node_out, node_in, inv_relation], dim=-1)
    edge_list = torch.stack([edge_list, inv_edge_list], dim=1).flatten(0, 1)
    num_nodes = torch.ones(num_rule, dtype=torch.long, device=solver.device) * 3
    num_edges = torch.ones(num_rule, dtype=torch.long, device=solver.device) * 4
    graph = data.PackedGraph(edge_list, num_nodes=num_nodes, num_edges=num_edges,
                             num_relation=_dataset.num_relation * 2)
    h_index = torch.zeros(num_rule, dtype=torch.long, device=solver.device)
    t_index = torch.ones(num_rule, dtype=torch.long, device=solver.device) * 2

    with torch.no_grad():
        score = solver.model.model(graph, h_index.unsqueeze(-1), t_index.unsqueeze(-1), rel_ac.unsqueeze(-1))
        score = score.squeeze(-1)
    f1 = 2 * coverage * confidence / (coverage + confidence)
    logger.warning("coverage - score spearmanr: %g" % metrics.spearmanr(coverage, score))
    logger.warning("confidence - score spearmanr: %g" % metrics.spearmanr(confidence, score))
    logger.warning("F1 - score spearmanr: %g" % metrics.spearmanr(f1, score))
    coverage = coverage.tolist()
    confidence = confidence.tolist()
    score = score.tolist()
    rel_ab = rel_ab.tolist()
    rel_bc = rel_bc.tolist()
    rel_ac = rel_ac.tolist()

    with open("/home/zhuzhaoc/fb15k237_nbfnet_rule.csv", "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["coverage", "confidence", "score", "relation ac", "relation ab", "relation bc"])
        for i in range(num_rule):
            writer.writerow([coverage[i], confidence[i], score[i],
                             relation_vocab[rel_ac[i]], relation_vocab[rel_ab[i]], relation_vocab[rel_bc[i]]])