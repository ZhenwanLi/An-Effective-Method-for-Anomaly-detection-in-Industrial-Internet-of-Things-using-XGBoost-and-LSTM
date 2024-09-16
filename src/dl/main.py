# code_first.src.dl.train.py
import sys
import os
import hydra
import numpy
import pandas
import seaborn
import sklearn
import torch
from omegaconf import DictConfig

import pyrootutils
from dl_test import Evaluator
from dl_train import Trainer

print("Python Path:", sys.path)
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def run_with_config(config_name: str):
    @hydra.main(version_base="1.3", config_path="configs", config_name=config_name)
    def inner_main(cfg: DictConfig):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.__version__)
        # print(hydra.__version__)
        # print(sklearn.__version__)
        # print(pandas.__version__)
        # print(numpy.__version__)
        # print(seaborn.__version__)
        if cfg.get("train"):
            trainer = Trainer(cfg, device)
            print(trainer.model)
            print(trainer.optimizer)
            print(trainer.scheduler)
            print(trainer.criterion)
            trainer.train()

        if cfg.get("test"):
            evaluator = Evaluator(cfg, device)
            print(evaluator.model)
            print(evaluator.criterion)
            evaluator.eval()
            evaluator.eval_tsne()

    inner_main()


if __name__ == "__main__":
    config_dir = "configs"
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]

    for config_file in config_files:
        config_name = config_file.replace(".yaml", "")
        print(f"Running with config: {config_name}")
        run_with_config(config_name)
