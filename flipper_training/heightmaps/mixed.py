from dataclasses import dataclass

import torch
from flipper_training.heightmaps import BaseHeightmapGenerator


@dataclass
class MixedHeightmapGenerator(BaseHeightmapGenerator):
    """
    Instantiates a list of heightmap generators and randomly selects one of them for each robot.
    The sampling  weights can be optionally specified.
    """

    classes: list[type[BaseHeightmapGenerator]]
    opts: list[dict]
    weights: list[float] | None = None

    def __post_init__(self):
        self.generators = [g(**o) for g, o in zip(self.classes, self.opts)]
        if self.weights is None:
            self.weights = [1.0 / len(self.generators)] * len(self.generators)
        self.weights_tensor = torch.tensor(self.weights)

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B, D, _ = x.shape
        z = torch.zeros_like(x)
        extras = {}
        zs, extras = list(zip(*[g._generate_heightmap(x, y, max_coord, rng) for g in self.generators]))
        selected_idx = torch.multinomial(self.weights_tensor, B, replacement=True)  # select
        # stack the Zs along the dim 1
        zs = torch.stack(zs, dim=1)
        # select the Zs according to the selected index
        z = zs[torch.arange(B), selected_idx]
        # Now we need to merge extras with overlapping names
        combined_extras = {}
        for i, extra1 in enumerate(extras):
            collected = {k: [] for k in extra1.keys()}  # create dicts for each index that has this key
            for j in range(i + 1, len(extras)):
                extra2 = extras[j]
                shared_keys = set(extra1.keys()).intersection(set(extra2.keys()))
                for key in shared_keys:
                    collected[key].append(j)  # remember the index
            # now unify them
            for key, indexlist in collected.items():
                combined_extras[key] = extra1[key]
                for index in indexlist:
                    selection_mask = selected_idx == index
                    combined_extras[key][selection_mask] = extras[index][key][selection_mask]
                    extras[index].pop(key)  # remove the key from the extras dict
        combined_extras["typelist"] = [type(self.generators[i]) for i in selected_idx]
        return z, combined_extras


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from flipper_training.heightmaps.multi_gaussian import MultiGaussianHeightmapGenerator
    from flipper_training.heightmaps.stairs import StairsHeightmapGenerator

    mix = MixedHeightmapGenerator(
        classes=[MultiGaussianHeightmapGenerator, StairsHeightmapGenerator],
        opts=[
            {"min_gaussians": 10, "max_gaussians": 20, "top_height_percentile_cutoff": 0.5},
            {"min_steps": 5, "max_steps": 6, "min_step_height": 0.1, "max_step_height": 0.3},
        ],
        weights=[0.5, 0.5],
    )

    res = 0.05
    max_coord = 3.2
    num = 16
    x, y, z, extras = mix(res, max_coord, num, rng=None)
    for i in range(num):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        ax[0].contourf(x[i], y[i], z[i], levels=100, cmap="viridis")
        ax[0].set_title("Heightmap")
        if extras["typelist"][i] == MultiGaussianHeightmapGenerator:
            ax[1].contourf(x[i], y[i], extras["suitable_mask"][i], levels=2, cmap="gray")
            ax[1].set_title("Suitable Mask")
        else:
            si = extras["step_indices"][i]
            ax[1].contourf(x[i], y[i], extras["step_indices"][i], levels=(si.max() - si.min() + 1).item(), cmap="gray")
            ax[1].set_title("Step Indices")
        plt.show()
