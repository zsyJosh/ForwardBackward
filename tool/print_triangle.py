import csv
from nltk.corpus import wordnet

from torchdrug import datasets

import model, util


util.setup_debug_hook()

vocab_file = "/home/b/bengioy/shiyu/kg-datasets/fb15k237_entity.txt"
target = "FB15k237"

def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]

    return entity_vocab, relation_vocab

if __name__ == "__main__":
    if target == "WN18RR":
        dataset = datasets.WN18RR("~/kg-datasets")
        offset2name = {}
        for s in wordnet.all_synsets():
            offset2name[s.offset()] = s.name()
        entity_vocab = [offset2name[int(e)] for e in dataset.entity_vocab]
        relation_vocab = dataset.relation_vocab + ["%s^{-1}" % r for r in dataset.relation_vocab]
    elif target == "FB15k237":
        dataset = datasets.FB15k237("~/kg-datasets")
        entity_vocab, relation_vocab = load_vocab(dataset)
        relation_vocab = relation_vocab + ["%s^{-1}" % r for r in relation_vocab]
    else:
        raise NotImplementedError

    # train graph
    graph = dataset.graph.edge_mask(slice(0, dataset.num_samples[0]))
    graph = graph.undirected(add_inverse=True)
    edge_ab, edge_bc, edge_ac = model.triangle_list(graph)
    a_index = graph.edge_list[edge_ab, 0].tolist()
    b_index = graph.edge_list[edge_ab, 1].tolist()
    c_index = graph.edge_list[edge_bc, 1].tolist()
    rel_ab = graph.edge_list[edge_ab, 2].tolist()
    rel_bc = graph.edge_list[edge_bc, 2].tolist()
    rel_ac = graph.edge_list[edge_ac, 2].tolist()

    with open("/home/b/bengioy/shiyu/scratch/%s_triangle.csv" % target.lower(), "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["entity a", "entity b", "entity c", "relation ac", "relation ab", "relation bc"])
        for i in range(len(edge_ab)):
            writer.writerow([entity_vocab[a_index[i]], entity_vocab[b_index[i]], entity_vocab[c_index[i]],
                             relation_vocab[rel_ac[i]], relation_vocab[rel_ab[i]], relation_vocab[rel_bc[i]]])
