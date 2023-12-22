from data_provider.data_loader import Dataset_RCA
from torch.utils.data import DataLoader

def data_provider(args, flag):
    Data = Dataset_RCA
    timeenc = 0
    train_only = False

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = 's'

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = 128
        freq = 's'

    data_set = Data(
        flag=flag,
        size=[32, 0, 1],
        features='MS',
        target="",
        timeenc=timeenc,
        freq=freq,
        train_only=train_only,
        root_path=args.root_path,
        data_path=args.data_path,
        scale=True
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader