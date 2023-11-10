import numpy as np
import torch
import sys
from transformers import BertTokenizer
from torch.utils.data import Dataset


One_hot = {'A': [1, 0, 0, 0],
           'T': [0, 1, 0, 0],
           'G': [0, 0, 1, 0],
           'C': [0, 0, 0, 1]}

NCP = {'A': [1, 1, 1],
       'T': [0, 1, 0],
       'G': [1, 0, 0],
       'C': [0, 0, 1]}

DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565, 0.5476708282666789],
        'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0, 0.76847598772376],
        'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764, 0.45903777008667923],
        'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944, 0.7467063799220581],
        'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811, 0.32686549630327577],
        'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565, 0.5476708282666789],
        'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195, 0.3501900760908136],
        'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026, 0.6891727614587756],
        'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026, 0.6891727614587756],
        'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944, 0.7467063799220581],
        'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261, 0.6083143907016332],
        'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172, 0.8400043540595654],
        'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195, 0.3501900760908136],
        'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764, 0.45903777008667923],
        'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868, 0.32686549630327577],
        'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261, 0.6083143907016332]}


class MyDataset(Dataset):
    def __init__(self, data, param):
        self.padding_length = param['max_padding_len']
        self.batch_size = param['batch_size']
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('./pretrain_model/dnabert')
        # self.tokenizer = BertTokenizer.from_pretrained('./pretrain_dna/checkpoint-22500')
        self.one_hot = One_hot
        self.NCP = NCP
        self.get_data()


    def rc_comp(self, seq):
        rc_dict = {"A": "T", "C": "G", "G": "C", "T": "A"}
        rc_seq = "".join([rc_dict[c] for c in seq[::-1]])
        return rc_seq


    def get_data(self):
        tmp_DNA_list = self.data.iloc[:, 1].values.tolist()
        tmp_DNA_list = [DNA[25:-25] for DNA in tmp_DNA_list]
        max_len = 50

        sen_list = []
        position_list = []

        # get physicochemical-embedding
        chains_one_hot = self.embedding_with_given_matrix(tmp_DNA_list, self.one_hot, max_len)
        chains_NCP = self.embedding_with_given_matrix(tmp_DNA_list, self.NCP, max_len)
        chains_DPCP = self.embedding_with_DPCP(tmp_DNA_list, max_len)

        phy_position = np.arange(50).reshape(1, 50, 1)
        phy_position = np.broadcast_to(phy_position, (len(tmp_DNA_list), 50, 1))
        # get bert-embedding
        for DNA in tmp_DNA_list:

            seq = []
            for i in range(0, len(DNA) - 5):
                seq.append(DNA[i:i + 6])
            input_ids = self.tokenizer.encode(seq, padding="max_length", max_length=47)

            position_ids = np.arange(len(input_ids))
            position_list.append(position_ids)
            sen_list.append(input_ids)

        self.sentence = np.array(sen_list)
        self.position = np.array(position_list)
        self.physico_embed = np.concatenate([chains_one_hot, chains_NCP, chains_DPCP, phy_position], axis=-1)
        self.labels = np.array(self.data.iloc[:, -3].values.tolist())

    def __getitem__(self, index):
        return torch.tensor(self.sentence[index], dtype=torch.long), torch.tensor(self.physico_embed[index],
                                                                                  dtype=torch.float), torch.tensor(
            self.position[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def embedding_with_given_matrix(self, aa_seqs, embedding_martix, max_seq_len):
        sequences = []

        for seq in aa_seqs:
            e_seq = np.zeros((len(seq), len(embedding_martix['A'])))

            for i, aa in enumerate(seq):
                if aa in embedding_martix:
                    e_seq[i] = embedding_martix[aa]
                else:
                    sys.stderr.write(
                        "Unknown amino acid in sequence: " + aa + ", encoding aborted!\n"
                    )
                    sys.exit(2)

            sequences.append(e_seq)

        num_seqs = len(aa_seqs)
        num_features = sequences[0].shape[1]

        embedded_aa_seq = np.zeros(
            (num_seqs, max_seq_len, num_features)
        )

        for i in range(0, num_seqs):
            embedded_aa_seq[i, : sequences[i].shape[0], :num_features] = sequences[i]

        return embedded_aa_seq

    def embedding_with_DPCP(self, aa_seqs, max_seq_len):
        sequences = []
        for i in range(len(aa_seqs)):
            sequence_cur = aa_seqs[i]

            e_seq = np.zeros([len(sequence_cur), 6])

            for pos in range(1, len(e_seq) - 1):
                e_seq[pos, :] += np.asarray(np.float32(DPCP[sequence_cur[pos:pos + 2]])) / 2
                e_seq[pos, :] += np.asarray(np.float32(DPCP[sequence_cur[pos - 1:pos + 1]])) / 2

            e_seq[0, :] = np.asarray(np.float32(DPCP[sequence_cur[0:2]]))
            e_seq[len(sequence_cur) - 1, :] = np.asarray(
                np.float32(DPCP[sequence_cur[len(sequence_cur) - 2:len(sequence_cur)]]))

            sequences.append(e_seq)

        num_seqs = len(aa_seqs)
        num_features = sequences[0].shape[1]

        embedded_aa_seq = np.zeros(
            (num_seqs, max_seq_len, num_features)
        )

        for i in range(0, num_seqs):
            embedded_aa_seq[i, : sequences[i].shape[0], :num_features] = sequences[i]

        return embedded_aa_seq
