import torch

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


######################## int16 -> int32 inference때 바꿈
def intersperse_exp(lst):
    min_ = min(lst)
    result = [min_] * (len(lst) * 2 + 1)
    tensor = torch.tensor(lst, dtype=torch.int32)
    result = torch.tensor(result, dtype=torch.int32)

    positive_mask = tensor > min_
    indices = torch.where(positive_mask)[0] * 2 + 1

    result[indices] = tensor[(indices - 1) // 2]

    left_indices = indices - 1
    result[left_indices] = tensor[(indices - 1) // 2]

    right_indices = indices + 1
    result[right_indices] = tensor[(indices - 1) // 2]
    return result

