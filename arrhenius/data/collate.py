from torch.utils.data import DataLoader
from chemprop.data.collate import collate_multicomponent as collate_mc


def build_loader_mc(
    dataset,
    batch_size=128,
    shuffle=False,
    generator=None,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mc,  # <-- this is the key line
        persistent_workers=persistent_workers,
    )
