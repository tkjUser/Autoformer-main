from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]    # args.data = 'custom' ==> Data = Dataset_Custom
          timeenc = 0 if args.embed != 'timeF' else 1  # 1

    if flag == 'test':           # test 时读取的数据
        shuffle_flag = False     # test 时候不打乱
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:                       # 训练和验证时候
        shuffle_flag = True     # 打乱数据
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Dataset_Custom
    data_set = Data(            # 实例化一个自定义的Dataset的对象
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],  # [96,48,96]
        features=args.features,  # M：输入多个变量，预测多个变量
        target=args.target,      # OT:目标特征
        timeenc=timeenc,         # args.embed = 'timeF'
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(     # torch.utils.data 中的DataLoader
        data_set,                 # Dataset_Custom
        batch_size=batch_size,    # 32
        shuffle=shuffle_flag,     # True（训练和验证时打乱数据）
        num_workers=args.num_workers,   # pycharm调试界面只支持主进程，线程开了之后就不能用了
        drop_last=drop_last)
    return data_set, data_loader   # 这里data_set 没必要返回
