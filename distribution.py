from plato.samplers import distribution_noniid
import torch
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler


class FixNonIID(distribution_noniid.Sampler):
    def __init__(self, datasource, client_id, testing):
        super().__init__(datasource, client_id, testing)
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        self.subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.client_partition_size,
                replacement=False,
                generator=gen,
            )
        )

    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)
