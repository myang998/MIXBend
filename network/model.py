import torch
import sys
import torch.nn.functional as F
from transformers import AutoModel
from torch import nn
from network.utils import MultiHeadAttention, SelfAttention
sys.path.append("./network")


def initialize_model(
        model_type,
        param,
        train=True
):
    model = MIXBend(param=param, train=train)
    return model


class MIXBend(nn.Module):
    def __init__(self, param, train=True):
        super(MIXBend, self).__init__()

        self.pos_encoder = AutoModel.from_pretrained('./pretrain_model/dnabert')

        self.conv1_in_channels = 14
        self.kernel_size = 6
        self.dropout = param["dropout"]
        self.embedding_dim = param["embedding_dim"]
        self.padding_len = 50

        self.flatten_size = self.padding_len - self.kernel_size + 1

        self.physico_attn = SelfAttention(dim=self.embedding_dim, attn_drop=self.dropout, proj_drop=self.dropout, output_attn=True)
        self.physico_encoder = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.conv1_in_channels, out_channels=self.embedding_dim,
                            kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.embedding_dim)

        if train:
            self.pos_encoder.train()

        self.pos_proj = nn.Linear(768, self.embedding_dim)
        self.physico_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.attn = MultiHeadAttention(n_head=param["n_head"], input_size=self.embedding_dim, hidden_size=self.embedding_dim, dropout=self.dropout, output_attn=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )

        self.temp = nn.Parameter(torch.ones([]) * 0.07)


    def forward(self, sent, physico_embed, position_ids):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        outputs = self.pos_encoder(sent, position_ids=position_ids, output_attentions=True)
        # 拿到last_hidden_state: (B, seq_dim, hidden_dim)
        embedding = outputs[0][:, 1:-1, :]
        # batch_size, 12, max_seq_length, max_seq_length
        attn_bert = outputs[-1][-1]

        conv_embed = self.physico_encoder(physico_embed.permute(0, 2, 1)).permute(0, 2, 1)
        # conv_embed = self.layer_norm(conv_embed)
        conv_embed, phy_attn = self.physico_attn(conv_embed)

        embedding = F.normalize(self.pos_proj(embedding), dim=-1)
        embedding_physico = F.normalize(self.physico_proj(conv_embed), dim=-1)


        ##======================== Contrastive ========================##
        # (batch, seq, seq)
        sim_o2r = torch.matmul(embedding, embedding_physico.permute(0, 2, 1)) / self.temp
        sim_r2d = torch.matmul(embedding_physico, embedding.permute(0, 2, 1)) / self.temp

        with torch.no_grad():
            # (batch, seq, seq)
            sim_targets = torch.zeros([sim_o2r.size(1), sim_o2r.size(2)])
            sim_targets.fill_diagonal_(1)
            sim_targets = sim_targets.unsqueeze(0).expand(sim_o2r.size(0), -1, -1)
            sim_targets = sim_targets.to(embedding.device)

        loss_o2r = -torch.mean(torch.sum(F.log_softmax(sim_o2r, dim=-1) * sim_targets, dim=-1), dim=1).mean()
        loss_r2o = -torch.mean(torch.sum(F.log_softmax(sim_r2d, dim=-1) * sim_targets, dim=-1), dim=1).mean()
        loss_contras = (loss_o2r + loss_r2o) / 2

        x = embedding + embedding_physico
        x, attn_concat = self.attn(x, x, x)

        x = torch.sum(x, 1)

        x = x.reshape((x.size(0), -1))
        x = F.normalize(x, dim=-1)
        logits = self.classifier(x)

        return logits, attn_bert, phy_attn, loss_contras
