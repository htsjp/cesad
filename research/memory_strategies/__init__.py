from . import duplicate_mean, fps_sampling, greedy_coreset, kmeans_sampling, random_sampling, region_mean


STRATEGIES = {
    "RegionMean_K1": region_mean.select,
    "DuplicateMean_K4": duplicate_mean.select,
    "Random_K4": random_sampling.select,
    "KMeans_K4": kmeans_sampling.select,
    "FPS_K4": fps_sampling.select,
    "GreedyCoreset_K4": greedy_coreset.select,
}
