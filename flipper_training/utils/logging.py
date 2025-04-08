import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from itertools import groupby
from queue import Queue
from typing import Any
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from flipper_training import ROOT

PROJECT = "flipper_training"


def red(s):
    return f"\033[91m{s}\033[00m"


def green(s):
    return f"\033[92m{s}\033[00m"


def yellow(s):
    return f"\033[93m{s}\033[00m"


def blue(s):
    return f"\033[94m{s}\033[00m"


def bold_red(s):
    return f"\033[31;1m{s}\033[00m"


@dataclass
class RunLogger:
    train_config: DictConfig
    use_wandb: bool
    category: str
    logfiles: dict = field(default_factory=dict)
    writers: dict = field(default_factory=dict)
    log_queue: Queue = field(default_factory=Queue)
    step_metric_name: str = "log_step"
    known_wandb_metrics: set = field(default_factory=set)

    def __post_init__(self):
        ts = time.strftime("%Y-%m-%d_%H:%M:%S")
        self.logpath = ROOT / f"runs/{self.category}/{self.train_config['name']}_{ts}"
        self.logpath.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.logpath / "weights"
        self.weights_path.mkdir(exist_ok=True)
        if self.use_wandb:
            wandb.init(
                project=PROJECT,
                name=f"{self.category}_{self.train_config['name']}_{time.strftime('%Y-%m-%d_%H:%M:%S')}",
                config=OmegaConf.to_container(self.train_config, resolve=False),
                save_code=True,
            )
            wandb.define_metric(self.step_metric_name)
        self._save_config()
        self.write_thread = threading.Thread(target=self._write, daemon=True)
        self.write_thread.start()

    def _save_config(self):
        OmegaConf.save(self.train_config, self.logpath / "config.yaml")

    def _init_logfile(self, name: str, sample_row: dict[str, Any]):
        self.logfiles[name] = open(self.logpath / f"{name}.csv", "w")
        writer = csv.DictWriter(self.logfiles[name], fieldnames=[self.step_metric_name] + list(sample_row.keys()))
        self.writers[name] = writer
        writer.writeheader()
        if self.use_wandb:
            for k in sample_row.keys():
                if k not in self.known_wandb_metrics:
                    wandb.define_metric(k, step_metric=self.step_metric_name)
                    self.known_wandb_metrics.add(k)
        return writer

    def log_data(self, row: dict[str, Any], step: int):
        self.log_queue.put((step, row))

    def _write_row(self, row: dict[str, Any], step: int):
        for topic, names in groupby(row.items(), key=lambda x: x[0].rsplit("/", maxsplit=1)[0]):
            topic_row = dict(names)
            writer = self.writers.get(topic, None) or self._init_logfile(topic, topic_row)
            writer.writerow(topic_row | {self.step_metric_name: step})

    def _write(self):
        while True:
            (step, row) = self.log_queue.get()
            if step == -1:
                break
            if self.use_wandb:
                wandb.log(data=row | {self.step_metric_name: step})
            self._write_row(row, step)

    def close(self):
        for f in self.logfiles.values():
            f.close()
        if self.use_wandb:
            wandb.finish()
        self.log_queue.put((-1, {}))
        self.write_thread.join()

    def save_weights(self, state_dict: dict, name: str):
        model_path = self.weights_path / f"{name}.pth"
        torch.save(state_dict, model_path)
        if self.use_wandb:
            wandb.log_model(
                path=model_path,
                name=name,
            )


@dataclass
class LocalRunReader:
    source: str | Path

    def __post_init__(self):
        self.path = Path(self.source).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        csvs = list(self.path.glob("*.csv"))
        print(f"Found available logs: {', '.join(map(str, csvs))}")

    def get_weights_path(self, name: str) -> Path:
        return self.path / "weights" / f"{name}.pth"

    def load_config(self) -> DictConfig:
        return OmegaConf.load(self.path / "config.yaml")


@dataclass
class WandbRunReader:
    run: str
    category: str
    default_weigths_path: Path = ROOT / "runs"

    def __post_init__(self):
        self.api = wandb.Api()
        self.run = self.api.run(f"{PROJECT}/{self.run}")
        self.history = self.run.scan_history()
        self.weights_root = self.default_weigths_path / f"{self.category}/{self.run}/wandb_weights"

    def get_weights_path(self, name: str) -> Path:
        self.run.file(name).download(self.weights_root)
        return self.weights_root / name

    def load_config(self) -> DictConfig:
        return OmegaConf.create(self.run.config)

    def get_metric(self, name: str) -> list:
        return [x[name] for x in self.history]


def get_run_reader(source: str, category: str) -> LocalRunReader | WandbRunReader:
    if source.startswith("wandb:"):
        return WandbRunReader(source.split(":")[1], category)
    return LocalRunReader(source)


class ColoredFormatter(logging.Formatter):
    base_fmt = "%(asctime)s [%(name)s][%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"

    def __init__(self):
        super().__init__()
        self.formatters = {}
        for fun, level in zip([blue, green, yellow, red, bold_red], [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]):
            level_fmt = self.base_fmt.replace("%(levelname)s", fun("%(levelname)s"))
            self.formatters[level] = logging.Formatter(level_fmt)

    def format(self, record):
        formatter = self.formatters.get(record.levelno)
        return formatter.format(record)


def get_terminal_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    return logger
