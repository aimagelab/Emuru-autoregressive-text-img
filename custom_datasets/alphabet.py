import torch

from .constants import (
    START_OF_SEQUENCE,
    END_OF_SEQUENCE,
    PAD,
)


class Alphabet:
    def __init__(self, charset):
        self.pad = PAD
        self.sos = START_OF_SEQUENCE
        self.eos = END_OF_SEQUENCE
        self.num_extra_tokens = 3
        self.extra_tokens = [self.pad, self.sos, self.eos]

        charset_types = [charset]

        self.char2idx = {}
        self.idx2char = {}
        current_id = self.num_extra_tokens
        for charset_type in charset_types:
            for char in charset_type:
                self.char2idx[char] = current_id
                self.idx2char[current_id] = char
                current_id += 1

    def __len__(self):
        return len(self.char2idx) + self.num_extra_tokens

    def encode(self, x_in):
        out = []
        for i in x_in:
            out.append(self.char2idx[i])
        return torch.LongTensor(out)

    def _decode(self, x_in, extra_tokens):
        out = []
        for i in x_in:
            if i not in extra_tokens:
                out.append(self.idx2char[int(i)])
        return "".join(out)

    def decode(self, x_in, stopping_logits: list):
        """Decode a batch of logits into a list of strings."""
        text = []
        for b in x_in:
            stops = []
            for s in stopping_logits:
                stop = torch.where(b == s)[0]
                if len(stop) == 0:
                    stop = torch.LongTensor([len(b)])
                stops.append(stop[0])
            end_idx = torch.min(torch.stack(stops))
            text.append(self._decode(b[:end_idx], self.extra_tokens))
        return text
