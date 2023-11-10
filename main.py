import argparse
import json
import logging
import gc
import pandas as pd
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dataset import MyDataset
from network.model import *
from train import *
from collections import OrderedDict
from transformers import get_linear_schedule_with_warmup
import matplotlib
matplotlib.use("agg")

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def implementation(args, param, X, fold, train_idx, test_idx):
    output_dir = f'ckpt/fold{fold}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{args.dataset}_{args.model}.pt')

    train_data = X.iloc[train_idx]
    test_data = X.iloc[test_idx]

    train_dataset = MyDataset(train_data, param)
    test_dataset = MyDataset(test_data, param)

    if args.local_rank != -1:
    # 2.DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=(train_sampler is None),
                                  sampler=train_sampler, pin_memory=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=param['batch_size']
        )
    print('train_data_finish')

    test_loader = DataLoader(test_dataset, batch_size=param['batch_size'], shuffle=False)
    print('test_data_finish')

    print(
        "Dataset: {}, Train set num: {}, Test set num: {}".format(
            args.dataset, len(train_dataset), len(test_dataset)
        )
    )

    model = initialize_model(
        model_type=args.model,
        param=param,
    )

    if args.load_pretrain:
        logging.info('loading pretrain parameters.')
        state_dict = torch.load('pretrain_model/clean_relu_dropout_128/pytorch_model.bin', map_location=torch.device(args.gpu_id))

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'bert' in k:
                start = k.index('bert')
                k = k[:start] + k[start + 5:]
            if 'cls' in k:
                continue
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    if args.local_rank != -1:
        # 3.创建DDP模型进行分布式训练
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
    else:
        model = model.to(args.gpu_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=param["lr"])
    t_total = len(train_loader) * param["epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * t_total), num_training_steps=t_total
    )

    criterion = nn.MSELoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        output_path=output_path,
        logger=logging.getLogger(__name__),
        local_rank=args.local_rank,
        gpu_id = args.gpu_id,
    )

    if args.local_rank in [-1, 0]:
        fold_train_loss, fold_eval_loss, fold_train_acc, fold_eval_acc, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr = trainer.train(
            train_loader=train_loader, test_loader=test_loader, epochs=param["epoch"], k=param["k"]
        )
        return fold_train_loss, fold_eval_loss, fold_train_acc, fold_eval_acc, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr
    else:
        trainer.train(
            train_loader=train_loader, test_loader=test_loader, epochs=param["epoch"], k=param["k"]
        )
        return


def parse_input(description: str) -> argparse.ArgumentParser:
    """ parsing input arguments
     """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="random",
        help="species of data, [random, nucleosomal]",
        type=str,
    )
    p.add_argument(
        "--model",
        default="MIXBend",
        required=False,
        help="Model type, [MIXBend]",
        type=str.lower,
    )
    p.add_argument(
        "--load_pretrain",
        default=True,
        required=False,
        type=bool
    )
    p.add_argument(
        "--gpu_id",
        # default="cuda",
        default="cuda:0",
        help="gpu id",
        type=str.lower,
    )
    p.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    p.add_argument(
        '--warmup_ratio',
        default=0.1,
        type=float,
        help='node rank for distributed training')
    return p.parse_args()


def main() -> None:
    args = parse_input("model")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    if args.local_rank != -1:
        # 1.初始化进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logging.info('use {} gpus!'.format(num_gpus))

    with open("./params/"+args.model+".json", "r") as f:
        param = json.load(f)
    for key, val in param.items():
        logging.info("argument %s: %r", key, val)
    for arg, value in sorted(vars(args).items()):
        logging.info("argument %s: %r", arg, value)

    dataset = f'data/{args.dataset}.txt'
    df = pd.read_table(dataset)
    df = df.sample(frac=1)

    X = df
    train_r2_result = []
    train_pearsonr_result = []
    train_separmanr_result = []
    test_r2_result = []
    test_pearsonr_result = []
    test_separmanr_result = []
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    skf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X)):
        print(f"----------------fold{fold} start.----------------")
        if args.local_rank in [-1, 0]:
            fold_train_loss, fold_eval_loss, fold_train_acc, fold_eval_acc, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr = implementation(args, param, X, fold, train_idx, test_idx)
            train_r2_result.append(train_r2)
            train_pearsonr_result.append(train_pearsonr)
            train_separmanr_result.append(train_separmanr)
            test_r2_result.append(test_r2)
            test_pearsonr_result.append(test_pearsonr)
            test_separmanr_result.append(test_separmanr)
            train_loss_list.append(fold_train_loss)
            test_loss_list.append(fold_eval_loss)
            train_acc_list.append(fold_train_acc)
            test_acc_list.append(fold_eval_acc)
        else:
            implementation(args, param, X, fold, train_idx, test_idx)

    if args.local_rank in [-1,0]:
        logging.info(
            "\nfinish!!\n train_r2: %.4f±%.4f, train_pearsonr: %.4f±%.4f \n train_separmanr: %.4f±%.4f, test_r2: %.4f±%.4f \n test_pearsonr: %.4f±%.4f, test_separmanr: %.4f±%.4f" % (
                np.mean(train_r2_result), np.std(train_r2_result), np.mean(train_pearsonr_result), np.std(train_pearsonr_result),
                np.mean(train_separmanr_result), np.std(train_separmanr_result), np.mean(test_r2_result), np.std(test_r2_result),
                np.mean(test_pearsonr_result), np.std(test_pearsonr_result), np.mean(test_separmanr_result), np.std(test_separmanr_result),
            )
        )


if __name__ == "__main__":
    main()
