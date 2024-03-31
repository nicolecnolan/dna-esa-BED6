from pathlib import Path

import torch

from dnaesa.config_schema import (ConfigSchema,
                                   DatasetConfigSchemaUniformSampling,
                                   SchedulerConfigSchema, TrainingConfigSchema)
from dnaesa.dataset import FastaUniformSampler
from dnaesa.main import main

device = torch.device("cuda:5")
CONFIG = ConfigSchema(
    training_config=TrainingConfigSchema(
        max_steps=100_000,
        batch_size=16,
        device=device,
        log_interval=100,
        accumulation_steps=16,
        scheduler_config=SchedulerConfigSchema(
            max_lr=1e-4,
        ),
    ),
    dataset_config=DatasetConfigSchemaUniformSampling(
        fasta_file = [Path("/mnt/SSD2/pholur/General_Models/data/all/chm13v2.0.fa")], 
        range_min = 800,
        range_max = 2000,
        subsequence_range_min = 80,
        subsequence_range_max = 180, 
        dataset=FastaUniformSampler,
        sampling_strategy="random_subsequence_uppercase",
        ),
)


main(CONFIG, watch_watch=True)
