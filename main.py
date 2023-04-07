import os
import math
import pprint
import shutil
import logging
import argparse

import torch
from torch.utils import data as torch_data

from torchdrug import core, utils, datasets, tasks, models
from torchdrug.utils import comm

import dataset
import layer
import model
import framework
import task
import engine
import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/knowledge_graph/wn18rr.yaml")
    parser.add_argument("-s", "--start", help="start config id for hyperparmeter search", type=int,
                        default=None)
    parser.add_argument("-e", "--end", help="end config id for hyperparmeter search", type=int,
                        default=None)

    return parser.parse_known_args()[0]


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


util.setup_debug_hook()
torch.manual_seed(1024 + comm.get_rank())

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfgs = util.load_config(args.config)

    output_dir = util.create_working_directory(cfgs[0])
    if comm.get_rank() == 0:
        logger = util.get_root_logger()

    start = args.start or 0
    end = args.end or len(cfgs)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning("Hyperparameter grid size: %d" % len(cfgs))
        logger.warning("Current job search range: [%d, %d)" % (start, end))
        shutil.copyfile(args.config, os.path.basename(args.config))

    cfgs = cfgs[start: end]
    for job_id, cfg in enumerate(cfgs):
        working_dir = output_dir
        if len(cfgs) > 1:
            working_dir = os.path.join(working_dir, str(job_id))
        if comm.get_rank() == 0:
            logger.warning("<<<<<<<<<< Job %d / %d start <<<<<<<<<<" % (job_id, len(cfgs)))
            logger.warning(pprint.pformat(cfg))
            os.makedirs(working_dir, exist_ok=True)
        comm.synchronize()
        os.chdir(working_dir)

        _dataset = core.Configurable.load_config_dict(cfg.dataset)
        train_set, valid_set, test_set = _dataset.split()

        small_train_set = torch_data.random_split(train_set, [len(valid_set), len(train_set) - len(valid_set)])[0]
        full_valid_set = valid_set
        if comm.get_rank() == 0:
            logger.warning(_dataset)
            logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

        if "fast_test" in cfg:
            if comm.get_rank() == 0:
                logger.warning("Quick test mode on. Only evaluate on %d samples for valid." % cfg.fast_test)
            g = torch.Generator()
            g.manual_seed(1024)
            valid_set = \
                torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
            # test_set = \
            #     torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]

        solver = build_solver(cfg)

        if "checkpoint" in cfg:
            solver.load(cfg.checkpoint)

        # train
        step = math.ceil(cfg.train.num_epoch / 10)
        best_score = float("-inf")
        best_epoch = -1

        if cfg.train.num_epoch > 0:
            for i in range(0, cfg.train.num_epoch, step):
                kwargs = cfg.train.copy()
                kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
                solver.train(**kwargs)
                solver.save("model_epoch_%d.pth" % solver.epoch)
                if isinstance(solver.model, tasks.KnowledgeGraphCompletion):
                    num_negative = solver.model.num_negative
                    solver.model.num_negative = _dataset.num_entity
                metric = solver.evaluate("valid")
                if isinstance(solver.model, tasks.KnowledgeGraphCompletion):
                    solver.model.num_negative = num_negative
                if "mrr" in metric:
                    score = metric["mrr"]
                else:
                    score = metric["accuracy"]
                if score > best_score:
                    best_score = score
                    best_epoch = solver.epoch

            solver.load("model_epoch_%d.pth" % best_epoch)

        # test
        solver.valid_set = full_valid_set
        if isinstance(solver.model, tasks.KnowledgeGraphCompletion):
            solver.model.num_negative = _dataset.num_entity
        solver.evaluate("valid")
        solver.evaluate("test")

        if comm.get_rank() == 0:
            logger.warning(">>>>>>>>>> Job %d / %d end >>>>>>>>>>" % (job_id, len(cfgs)))
