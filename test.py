import argparse
import logging
import json
import pandas as pd
from tqdm import tqdm
from dataset import *
from network.model import *
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def test_independent(pred_dataloader, model, device="cpu"):
    model.to(device)
    model.eval()

    epoch_targets = []
    epoch_preds = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(pred_dataloader)):
            torch.cuda.empty_cache()
            sentence = data[0].to(device)
            reverse = data[1].to(device)
            ids = data[2].to(device)
            iter_target = data[3].to(device)
            iter_output, _, contras_loss = model(sentence, reverse, ids)
            epoch_targets.extend(iter_target)
            epoch_preds.extend(iter_output)

    epoch_targets = torch.tensor(epoch_targets).detach().cpu().numpy()
    epoch_preds = torch.tensor(epoch_preds).detach().cpu().numpy()

    epoch_r2 = r2_score(epoch_targets, epoch_preds)
    epoch_pearsonr = pearsonr(epoch_targets, epoch_preds)
    epoch_spearmanr = spearmanr(epoch_targets, epoch_preds)

    return epoch_r2, epoch_pearsonr[0], epoch_spearmanr[0], epoch_preds

def parse_input(description: str) -> argparse.ArgumentParser:
    """ parsing input arguments
     """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--training_dataset",
        default="random",
        help="species of data, [random, nucleosomal]",
        type=str,
    )
    p.add_argument(
        "--testing_dataset",
        default="tiling",
        help="species of data, [chrV, tiling]",
        type=str,
    )
    p.add_argument(
        "--model",
        default="MIXBend",
        required=False,
        help="Model type, [nn1, concat, dnabert]",
        type=str.lower,
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        required=False,
        help="cpu, cuda",
        type=str.lower,
    )
    return p.parse_args()


def main():
    args = parse_input("model")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    with open("./params/" + args.model + ".json", "r") as f:
        param = json.load(f)
    for key, val in param.items():
        logging.info("argument %s: %r", key, val)
    for arg, value in sorted(vars(args).items()):
        logging.info("argument %s: %r", arg, value)

    test_data_path = f"data/{args.testing_dataset}.txt"
    test_r2_result = []
    test_pearsonr_result = []
    test_separmanr_result = []
    test_pred_result = []
    df = pd.read_table(test_data_path)
    test_dataset = MyDataset(df, param)

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    test_loader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=param['batch_size']
    )

    logging.info(
        "Dataset: {}, Test set num: {}".format(
            args.testing_dataset, len(test_dataset)
        )
    )

    for i in range(10):
        print('fold ', i)
        embed_output_dir = f"ckpt/fold{i}/"

        model = initialize_model(
            model_type=args.model,
            param=param,
            train=False
        )

        if args.device == "cpu":
            model.load_state_dict(torch.load(embed_output_dir + args.training_dataset + '_' + args.model + '.pt', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(embed_output_dir + args.training_dataset + '_' + args.model + '.pt'))

        epoch_r2, epoch_pearsonr, epoch_spearmanr, epoch_preds = test_independent(test_loader, model, device=args.device)
        test_r2_result.append(epoch_r2)
        test_pearsonr_result.append(epoch_pearsonr)
        test_separmanr_result.append(epoch_spearmanr)
        test_pred_result.append(epoch_preds)
        logging.info(
            "test_r2: {:.6f}, test_pearsonr : {:.6f}, test_separmanr: {:.6f}".format(
                epoch_r2, epoch_pearsonr, epoch_spearmanr
            )
        )
    logging.info(
        "\nfinish!!\n test_r2: %.4f±%.4f \n test_pearsonr: %.4f±%.4f, test_separmanr: %.4f±%.4f" % (
            np.mean(test_r2_result), np.std(test_r2_result), np.mean(test_pearsonr_result), np.std(test_pearsonr_result), np.mean(test_separmanr_result), np.std(test_separmanr_result)
        )
    )

    test_pred_result = np.array(test_pred_result)
    test_pred_result = np.mean(test_pred_result, axis=0)
    print(test_pred_result.shape)
    df['pred_c0'] = test_pred_result
    df.to_csv(f"data/{args.training_dataset}_{args.testing_dataset}_pred.txt", index=False, sep='\t')

if __name__ == "__main__":
    main()
