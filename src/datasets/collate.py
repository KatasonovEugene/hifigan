import torch


def train_collate_fn(dataset_items: list[dict]):
    result_batch = {}
    result_batch["gt_audio"] = torch.nn.utils.rnn.pad_sequence(
        [item["gt_audio"] for item in dataset_items],
        batch_first=True,
    )
    result_batch["gt_melspec"] = result_batch["gt_audio"].clone()
    result_batch["gt_audio"] = result_batch["gt_audio"].unsqueeze(1)
    result_batch["sample_rate"] = dataset_items[0]["sample_rate"]
    return result_batch


def inference_collate_fn(dataset_items: list[dict]):
    result_batch = {}
    if "gt_audio" in dataset_items[0].keys():
        result_batch["gt_audio"] = torch.nn.utils.rnn.pad_sequence(
            [item["gt_audio"] for item in dataset_items],
            batch_first=True,
        )
        result_batch["gt_melspec"] = result_batch["gt_audio"].clone()
        result_batch["gt_audio"] = result_batch["gt_audio"].unsqueeze(1)
    result_batch["sample_rate"] = dataset_items[0]["sample_rate"]
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["text_filename"] = [item["text_filename"] for item in dataset_items]
    return result_batch
