import json
import os
import numpy as np
import torch



def sample_indices_to_drop(dataset: str, drop_criterion: str, indices_class, 
                 keep_percentile_from, keep_percentile_to, 
                 CRITERION_ROOT:str = "./drop_utils/resource"):
    """Parameters:
    dataset: (str) dataset name
    drop_criterion: (str) =`random`, or in the format of ${utility-indicator}_${order}, e.g. LossConverge_Small
    keep_percentile_from, keep_percentile_to: (float) keep the samples in this range. We set two params for the stratified experiment.
    Return:
    dropped_idx_set: (set[int]) the sample indices to remove"""
    
    # find samples to remove
    dropped_idx_set = set()

    if drop_criterion.lower() in ["random", "rand"]:
        print("Randomly drop")
        for c, idx_group in enumerate(indices_class):
            keep_from = int(keep_percentile_from * len(idx_group))
            keep_to   = int(keep_percentile_to * len(idx_group))
            np.random.shuffle(idx_group)
            drop_ids = idx_group[:keep_from] + idx_group[keep_to:]
            dropped_idx_set.update(drop_ids)
    
    else:
        try:
            utility_indicator, drop_order = drop_criterion.split("_")
        except:
            raise ValueError(drop_criterion)
        
        print(f"Drop according to {utility_indicator} with {drop_order} values")
        drop_order = drop_order.lower()
        assert drop_order in ['large', 'small']
        
        # load from existing json file
        score_file = os.path.join(CRITERION_ROOT, f"{dataset}_{utility_indicator}.json")
        assert os.path.exists(score_file), score_file
        with open(score_file) as fp:
            utility_values = np.array(json.load(fp))
        
        if drop_order == "large":
            utility_values = -utility_values
        # We reverse the score if drop_order==large, 
        #   so that in both cases we could simply drop the samples at the left of the sorted array.

        for c, idx_group in enumerate(indices_class):
            keep_from = int(keep_percentile_from * len(idx_group))
            keep_to   = int(keep_percentile_to * len(idx_group))
            idx_group = sorted(idx_group, key=lambda i:utility_values[i])
            drop_ids = idx_group[:keep_from] + idx_group[keep_to:]  # 默认order=small，扔小的
            dropped_idx_set.update(drop_ids)
    
    return dropped_idx_set

        





def drop_samples(images_all, labels_all, indices_class,
                dataset: str, drop_criterion: str, 
                 *, drop_ratio=None, keep_ratio=None):
    """images_all, labels_all, indices_class: the dataset structure that commonly used for DD
    dataset: (str) dataset name
    drop_criterion: (str) =`random`, or in the format of ${utility-indicator}_${order}, e.g. LossConverge_Small
    drop_ratio, keep_ratio: only one of them should be specified (drop_ratio = 1.0 - keep_ratio)
    """
    assert (drop_ratio is None) ^ (keep_ratio is None), \
            f"Only one of drop_ratio ({drop_ratio}) and keep_ratio ({keep_ratio}) should be specified."
    
    if drop_ratio is None:
        assert keep_ratio is not None, "I know keep_ratio must have value here! I'm muting the warning in my way."
        drop_ratio = 1.0 - keep_ratio
    assert 0.0 <= drop_ratio <= 1.0, str(drop_ratio)

    # Here's the tricky part: remember that in any case, the samples we hope to drop is sorted to the left
    #     of the sequence, so we keep the `keep_ratio`% samples at right, 
    #     i.e. we keep the range [drop_ratio, 100%]
    
    dropped_idx_set = sample_indices_to_drop(dataset, drop_criterion, indices_class, drop_ratio, 1.0)


    # re-indexing
    
    images_all = [x for i, x in enumerate(images_all) if i not in dropped_idx_set]
    print("Original:", labels_all.shape[0], "; Now:", len(images_all), "remain")
    labels_all = [x for i, x in enumerate(labels_all) if i not in dropped_idx_set]

    indices_class = [[] for c in range(len(indices_class))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    # for i, x in enumerate(indices_class):
    #     print("Class", i, "remains", len(x), "samples")

    images_all = torch.stack(images_all, dim=0)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=images_all.device)
    torch.cuda.empty_cache()

    return images_all, labels_all, indices_class