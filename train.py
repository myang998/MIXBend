import torch
import logging
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        output_path,
        logger=logging.getLogger(__name__),
        local_rank=-1,
        gpu_id="cuda:0",
        is_progress_bar=True,
    ):
        if local_rank == -1:
            self.device = torch.device(gpu_id)
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        self.model = model
        self.local_rank = local_rank
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_path = output_path

        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.logger.info("Training Device: {}".format(self.device))

    def train(self, train_loader, test_loader, epochs=100, k=0.5):
        train_loss, test_loss, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        train_loss_list = []
        train_acc_list = []
        eval_loss_list = []
        eval_acc_list = []

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_r2, train_pearsonr, train_separmanr = self._train_epoch(train_loader, k)
            test_loss, test_r2, test_pearsonr, test_separmanr = self._validate_epoch(test_loader)

            if self.local_rank in [-1,0]:
                self.logger.info(
                    "Epoch: {} Average training loss : {:.6f}, average_test_loss : {:.6f}, train_r2: {:.6f}, train_pearsonr: {:.6f}, train_separmanr: {:.6f}, test_r2: {:.6f}, test_pearsonr : {:.6f}, test_separmanr: {:.6f}".format(
                        epoch + 1, train_loss, test_loss, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr
                    )
                )
                # save testint result...
                train_loss_list.append(train_loss)
                train_acc_list.append(train_pearsonr)
                eval_loss_list.append(test_loss)
                eval_acc_list.append(test_pearsonr)

        torch.save(self.model.state_dict(), self.output_path)
        self.logger.info("Saving model checkpoint to %s", self.output_path)

        return train_loss_list, eval_loss_list, train_acc_list, eval_acc_list, train_r2, train_pearsonr, train_separmanr, test_r2, test_pearsonr, test_separmanr


    def _train_epoch(self, train_loader, k):
        epoch_loss = 0.0
        epoch_targets = []
        epoch_preds = []

        for idx, data in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()
            sentence = data[0].to(self.device)
            reverse = data[1].to(self.device)
            ids = data[2].to(self.device)
            iter_target = data[3].to(self.device)
            iter_output, _, _, contras_loss = self.model(sentence, reverse, ids)
            iter_loss = self.criterion(iter_output.squeeze(), iter_target.squeeze())
            total_loss = iter_loss + k * contras_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += iter_loss.item()

            epoch_targets.extend(iter_target)
            epoch_preds.extend(iter_output)

        mean_epoch_loss = epoch_loss / len(train_loader)

        epoch_targets = torch.tensor(epoch_targets).detach().cpu().numpy()
        epoch_preds = torch.tensor(epoch_preds).detach().cpu().numpy()


        epoch_r2 = r2_score(epoch_targets, epoch_preds)
        epoch_pearsonr = pearsonr(epoch_targets, epoch_preds)
        epoch_spearmanr = spearmanr(epoch_targets, epoch_preds)

        return mean_epoch_loss, epoch_r2, epoch_pearsonr[0], epoch_spearmanr[0]

    def _validate_epoch(self, valid_loader):
        self.model.eval()

        with torch.no_grad():
            val_loss = 0
            val_targets = []
            val_preds = []

            for data in valid_loader:
                torch.cuda.empty_cache()
                sentence = data[0].to(self.device)
                reverse = data[1].to(self.device)
                ids = data[2].to(self.device)
                target = data[3].to(self.device)
                output, _, _, _ = self.model(sentence, reverse, ids)
                val_preds.extend(output)
                val_targets.extend(target)

                loss = self.criterion(output.squeeze(), target.squeeze())
                val_loss += loss.item()

        val_targets = torch.tensor(val_targets).detach().cpu().numpy()
        val_preds = torch.tensor(val_preds).detach().cpu().numpy()

        mean_loss = val_loss / len(valid_loader)

        epoch_r2 = r2_score(val_targets, val_preds)
        epoch_pearsonr = pearsonr(val_targets, val_preds)
        epoch_spearmanr = spearmanr(val_targets, val_preds)

        return mean_loss, epoch_r2, epoch_pearsonr[0], epoch_spearmanr[0]
