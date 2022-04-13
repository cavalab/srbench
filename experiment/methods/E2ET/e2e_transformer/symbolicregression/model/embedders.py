from typing import Tuple, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from symbolicregression.utils import to_cuda
import torch.nn.functional as F

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )
        self.float_scalar_descriptor_len = (2 + self.params.mantissa_len)
        self.total_dimension = self.params.max_input_dimension + self.params.max_output_dimension
        self.float_vector_descriptor_len = self.float_scalar_descriptor_len * self.total_dimension

        self.activation_fn = F.relu
        size = self.float_vector_descriptor_len*self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_len

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)

        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers: sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings

    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.params.cpu)
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                x_toks = [
                    *x_toks,
                    *[
                        "<INPUT_PAD>"
                        for _ in range(
                            (self.params.max_input_dimension - input_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                y_toks = [
                    *y_toks,
                    *[
                        "<OUTPUT_PAD>"
                        for _ in range(
                            (self.params.max_output_dimension - output_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs, self.float_vector_descriptor_len).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths


class TNetEmbedder(Embedder):
    def __init__(self, params, env):
        super().__init__()
        self.env = env
        self.params = params
        self.pad_to_max_dim = params.pad_to_max_dim
        self.max_seq_len = 1
        self.dim = params.enc_emb_dim

        self.use_batch_norm = False
        self.input_batch_norm = nn.BatchNorm2d(1)
        self.activation_fn = F.relu
        self.fc1 = nn.Linear(self.params.max_input_dimension + self.params.max_output_dimension, self.dim)
        self.fc2 = nn.Linear(self.dim, 2*self.dim)
        self.fc3 = nn.Linear(2*self.dim, 4*self.dim)
        self.fc4 = nn.Linear(4*self.dim, 2*self.dim)
        self.fc5 = nn.Linear(2*self.dim, self.dim)

    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(
            sequences, sequences_len, use_cpu=self.params.cpu
        )
        sequences_embeddings = self.embed(sequences)
        return sequences_embeddings, torch.ones_like(sequences_len)

    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = x
                y_toks = y
                input_dim = len(x_toks)
                output_dim = len(y_toks)
                x_toks = [
                    *x_toks,
                    *[
                        0.0
                        for _ in range(
                            (self.params.max_input_dimension - input_dim)
                        )
                    ],
                ]
                y_toks = [
                    *y_toks,
                    *[
                        0.0
                        for _ in range(
                            (self.params.max_output_dimension - output_dim)
                        )
                    ],
                ]
                toks = [*x_toks, *y_toks]
                seq_toks.append(toks)
            res.append(torch.FloatTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.FloatTensor(slen, bs, self.params.max_input_dimension + self.params.max_output_dimension).fill_(0.0)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.transpose(0, 1).unsqueeze(1)
        B = batch.shape[0]

        x = self.input_batch_norm(batch).squeeze(1) if self.use_batch_norm else batch.squeeze(1)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        assert x.shape[-1] == 4*self.dim
        x, _ = torch.max(x, dim=1)  # global max pooling
        x = self.activation_fn(self.fc4(x))
        x = self.activation_fn(self.fc5(x))
        x = x.unsqueeze(0)
        assert x.shape[0] == 1 and x.shape[1] == B and x.shape[2] == self.dim
        return x

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.ones(len(seqs), dtype=torch.long)
        return lengths

class FlatEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.pad_to_max_dim = params.pad_to_max_dim
        sep = 0 if self.pad_to_max_dim else 2
        self.max_seq_len = self.params.max_len * (
            (2 + self.params.mantissa_len)
            * (self.params.max_input_dimension + self.params.max_output_dimension)
            + sep
        )
        self.dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )

    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(
            sequences, sequences_len, use_cpu=self.params.cpu
        )
        sequences_embeddings = self.embed(sequences)
        return sequences_embeddings, sequences_len

    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                if self.pad_to_max_dim:
                    input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                    output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                    x_toks = [
                        *x_toks,
                        *[
                            "<INPUT_PAD>"
                            for _ in range(
                                (self.params.max_input_dimension - input_dim)
                                * (2 + self.params.mantissa_len)
                            )
                        ],
                    ]
                    y_toks = [
                        *y_toks,
                        *[
                            "<OUTPUT_PAD>"
                            for _ in range(
                                (self.params.max_output_dimension - output_dim)
                                * (2 + self.params.mantissa_len)
                            )
                        ],
                    ]
                    toks = [*x_toks, *y_toks]
                else:
                    toks = [*x_toks, "</X>", *y_toks, "</Y>"]
                seq_toks.extend([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            if self.pad_to_max_dim:
                sep, d_in, d_out = (
                    0,
                    self.params.max_input_dimension,
                    self.params.max_output_dimension,
                )
            else:
                x, y = seq[0]
                sep, d_in, d_out = 2, len(x), len(y)
            lengths[i] = len(seq) * (
                (2 + self.params.mantissa_len) * (d_in + d_out) + sep
            )
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths


class AttentionPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding, MultiHeadAttention, TransformerFFN

        super().__init__()
        self.env = env
        self.params = params
        self.dim = params.enc_emb_dim
        self.pad_to_max_dim = params.pad_to_max_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )
        self.float_scalar_descriptor_len = (2 + self.params.mantissa_len)
        self.total_dimension = self.params.max_input_dimension + self.params.max_output_dimension
        self.float_vector_descriptor_len = self.float_scalar_descriptor_len * self.total_dimension

        self.activation_fn = F.relu
        self.attn = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(self.params.n_emb_layers):
            self.attn.append(MultiHeadAttention(self.params.n_enc_heads, self.dim, self.dim, self.params.attention_dropout, self.params.norm_attention))
            self.norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.ffn.append(TransformerFFN(self.dim, self.dim*4, self.dim, self.params.n_enc_hidden_layers, dropout=self.params.dropout))
            self.norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        
        if self.pad_to_max_dim:
            self.fc = nn.Linear(self.float_vector_descriptor_len*self.dim, self.dim)
        self.max_seq_len = self.params.max_len

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)

        """
        bs, max_len, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.reshape(bs,-1,dim).permute(1,0,2)
        for i in range(self.params.n_emb_layers):
            sequences_embeddings = sequences_embeddings + self.attn[i](sequences_embeddings)
            sequences_embeddings = self.norm1[i](sequences_embeddings)
            sequences_embeddings = sequences_embeddings + self.ffn[i](sequences_embeddings)
            sequences_embeddings = self.norm2[i](sequences_embeddings)
        if self.pad_to_max_dim:
            sequences_embeddings = sequences_embeddings.permute(1,0,2).reshape(bs,max_len,-1)
            sequences_embeddings = self.activation_fn(self.fc(sequences_embeddings))
        else:
            sequences_embeddings = sequences_embeddings.permute(1,0,2).reshape(bs, max_len, float_descriptor_length, dim)[:,:,0]
        return sequences_embeddings

    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.params.cpu)
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                if self.pad_to_max_dim:
                    x_pad_len = (self.params.max_input_dimension - input_dim) * self.float_scalar_descriptor_len
                    y_pad_len= (self.params.max_output_dimension - output_dim) * self.float_scalar_descriptor_len
                    x_toks = [*x_toks,*["<INPUT_PAD>" for _ in range(x_pad_len)]]
                    y_toks = [*y_toks,*["<OUTPUT_PAD>" for _ in range(y_pad_len)]]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths, descriptor_lengths = [x.size(0) for x in seqs], [x.size(1) for x in seqs]
        bs, slen, dlen = len(lengths), max(lengths), max(descriptor_lengths)
        sent = torch.LongTensor(slen, bs, dlen).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[:seq.size(0), i, :seq.size(1)] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths

    
##TODO: ConvPoint is equivalent to LinearPoint now, will change it to be able to do more fancy stuff with convolutions
class ConvPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.dim = params.enc_emb_dim
        self.max_seq_len = self.params.max_len
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )

        self.point_len = (2 + self.params.mantissa_len) * (
            self.params.max_input_dimension + self.params.max_output_dimension
        )
        # can either convolve the entire point (i.e. in 2d: in the dimension axis and per-dimension embedding axis)
        # using multiple kernels of large size (+ large stride)
        # OR convolve over the dimension axis only, using small kernel size/stride and one output channel
        ##or can use
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,  # self.dim,
            kernel_size=(self.point_len, 1),  # self.dim),
            stride=(self.point_len, 1),
        )  # self.dim))

    def compress(
        self, sequences_embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)

        """
        lengths = (
            lengths
            / (2 + self.params.mantissa_len)
            / (self.params.max_input_dimension + self.params.max_output_dimension)
        )
        lengths = lengths.long()
        sequences_embeddings = sequences_embeddings.transpose(0, 1).unsqueeze(1)
        bs, _, max_len, dim = sequences_embeddings.size()
        compressed_embeddings = self.conv(sequences_embeddings)
        compressed_embeddings = compressed_embeddings.squeeze(1)
        compressed_embeddings = compressed_embeddings.transpose(0, 1)
        # compressed_embeddings = compressed_embeddings.squeeze(-1)
        # compressed_embeddings = compressed_embeddings.transpose(0,2)
        # compressed_embeddings = compressed_embeddings.transpose(1,2)
        _max_len, _bs, _dim = compressed_embeddings.size()
        assert bs == _bs and _max_len * self.point_len == max_len and _dim == dim
        return compressed_embeddings, lengths

    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(
            sequences, sequences_len, use_cpu=self.params.cpu
        )
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings, sequences_len = self.compress(
            sequences_embeddings, sequences_len
        )
        return sequences_embeddings, sequences_len

    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                x_toks = [
                    *x_toks,
                    *[
                        "<INPUT_PAD>"
                        for _ in range(
                            (self.params.max_input_dimension - input_dim)
                            * (2 + self.params.mantissa_len)
                        )
                    ],
                ]
                y_toks = [
                    *y_toks,
                    *[
                        "<OUTPUT_PAD>"
                        for _ in range(
                            (self.params.max_output_dimension - output_dim)
                            * (2 + self.params.mantissa_len)
                        )
                    ],
                ]
                toks = [*x_toks, *y_toks]
                seq_toks.extend([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len
        return lengths

if __name__ == "__main__":
    from symbolicregression.model import build_modules
    from symbolicregression.envs import build_env
    from train import get_parser
    import numpy as np

    parser = get_parser()
    params = parser.parse_args()
    args = params
    args.embedder_type = "LinearPoint"#"LinearPoint"

    env = build_env(args)
    env.rng = np.random.RandomState(0)

    modules = build_modules(env, args)
    embedder = modules["embedder"]
    x = np.random.randn(10, 1)
    y = np.expand_dims(np.cos(x[:, 0]), -1)
    seq = [(x[i], y[i]) for i in range(x.shape[0])]

    x2 = np.expand_dims(np.arange(5),-1)
    y2 = np.expand_dims(np.sin(x[:, 0]), -1)
    seq2 = [(x2[i], y2[i]) for i in range(x2.shape[0])]
    seqs = [seq, seq2]
    print(embedder.get_length_after_batching(seqs))
    encoded = embedder.encode(seqs)
    embeddings = embedder([seq, seq2])
    print(embeddings[0].shape, embeddings[1])
