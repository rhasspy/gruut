"""
Credit: kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
"""
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import PriorityQueue

import grapheme
import numpy as np

# -----------------------------------------------------------------------------


class FixedSymbols(str, Enum):
    """Pseudo-graphemes with fixed indexes"""

    PAD = "<pad>"
    SOS = "<s>"
    EOS = "</s>"


@dataclass
class BeamSearchItem:
    """Item in predict beam search"""

    dec: np.ndarray
    h: np.ndarray
    phoneme_id: int
    log_prob: float = 0
    length: int = 1
    parent: "typing.Optional[BeamSearchItem]" = None

    @property
    def score(self):
        """Score used for priority queue"""
        return self.log_prob


# -----------------------------------------------------------------------------


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _grucell(x, h, w_ih, w_hh, b_ih, b_hh):
    rzn_ih = np.matmul(x, w_ih.T) + b_ih
    rzn_hh = np.matmul(h, w_hh.T) + b_hh

    rz_ih, n_ih = (
        rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3],
        rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :],
    )
    rz_hh, n_hh = (
        rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3],
        rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :],
    )

    rz = _sigmoid(rz_ih + rz_hh)
    r, z = np.split(rz, 2, -1)  # pylint: disable=unbalanced-tuple-unpacking

    n = np.tanh(n_ih + r * n_hh)
    h = (1 - z) * n + z * h

    return h


def _gru(x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
    if h0 is None:
        h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
    h = h0  # initial hidden state
    outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
    for t in range(steps):
        h = _grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
        outputs[:, t, ::] = h
    return outputs


# -----------------------------------------------------------------------------


class GeepersG2P:
    """Phoneme prediction from a pre-trained model using pure numpy"""

    def __init__(
        self,
        graphemes: typing.List[str],
        phonemes: typing.List[str],
        dec_maxlen: int = 20,
    ):
        self.graphemes = [FixedSymbols.PAD.value, FixedSymbols.EOS.value] + graphemes
        self.phonemes = [
            FixedSymbols.PAD.value,
            FixedSymbols.EOS.value,
            FixedSymbols.SOS.value,
        ] + phonemes

        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}

        self.pad_idx = 0
        self.eos_idx = 1
        self.sos_idx = 2

        self.dec_maxlen = dec_maxlen

        # Empty until load_variables is called
        self.enc_emb = np.empty((1,))
        self.enc_w_ih = np.empty((1,))
        self.enc_w_hh = np.empty((1,))
        self.enc_b_ih = np.empty((1,))
        self.enc_b_hh = np.empty((1,))

        self.dec_emb = np.empty((1,))
        self.dec_w_ih = np.empty((1,))
        self.dec_w_hh = np.empty((1,))
        self.dec_b_ih = np.empty((1,))
        self.dec_b_hh = np.empty((1,))
        self.fc_w = np.empty((1,))
        self.fc_b = np.empty((1,))

    def load_variables(self, npz_path: typing.Union[str, Path]):
        """Load encoder/decoder weights from .npz file"""
        variables = np.load(str(npz_path))
        self.enc_emb = variables["encoder.emb.weight"]  # (len(graphemes), emb)
        self.enc_w_ih = variables["encoder.rnn.weight_ih_l0"]  # (3*128, 64)
        self.enc_w_hh = variables["encoder.rnn.weight_hh_l0"]  # (3*128, 128)
        self.enc_b_ih = variables["encoder.rnn.bias_ih_l0"]  # (3*128,)
        self.enc_b_hh = variables["encoder.rnn.bias_hh_l0"]  # (3*128,)

        self.dec_emb = variables["decoder.emb.weight"]  # (len(phonemes), emb)
        self.dec_w_ih = variables["decoder.rnn.weight_ih_l0"]  # (3*128, 64)
        self.dec_w_hh = variables["decoder.rnn.weight_hh_l0"]  # (3*128, 128)
        self.dec_b_ih = variables["decoder.rnn.bias_ih_l0"]  # (3*128,)
        self.dec_b_hh = variables["decoder.rnn.bias_hh_l0"]  # (3*128,)
        self.fc_w = variables["decoder.fc.weight"]  # (74, 128)
        self.fc_b = variables["decoder.fc.bias"]  # (74,)

    def predict(
        self, word: str, num_guesses: int = 1, beam_width: int = 0
    ) -> typing.List[typing.List[str]]:
        """Predict phonemes for the given word"""
        # encoder
        graphemes = list(grapheme.graphemes(word))
        enc = self._encode(graphemes)
        enc = _gru(
            enc,
            len(graphemes) + 1,
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32),
        )

        if beam_width <= 0:
            return self._predict_greedy(enc)

        return self._predict_beam_search(
            enc, beam_width=beam_width, num_guesses=num_guesses
        )

    # -------------------------------------------------------------------------

    def _encode(self, graphemes: typing.List[str]) -> np.ndarray:
        x = [self.g2idx[g] for g in graphemes] + [self.eos_idx]
        return np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

    def _predict_greedy(self, enc: np.ndarray) -> typing.List[typing.List[str]]:
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [self.sos_idx], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for _ in range(self.dec_maxlen):
            h = _grucell(
                dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh
            )  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == self.eos_idx:
                break  # </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.phonemes[idx] for idx in preds]
        return [preds]

    def _predict_beam_search(
        self,
        enc: np.ndarray,
        beam_width: int = 10,
        num_guesses: int = 10,
        max_steps: int = 2000,
    ) -> typing.List[typing.List[str]]:
        start_hidden = enc[:, -1, :]

        # decoder
        start_dec = typing.cast(
            np.ndarray, np.take(self.dec_emb, [self.sos_idx], axis=0)
        )  # 2: <s>

        start_item = BeamSearchItem(
            dec=start_dec, h=start_hidden, phoneme_id=self.sos_idx
        )

        q: "PriorityQueue[typing.Tuple[float, BeamSearchItem]]" = PriorityQueue()
        q.put((0.0, start_item))

        num_steps: int = 0
        completed_items: typing.List[typing.Tuple[float, typing.List[str]]] = []

        while not q.empty():
            num_steps += 1
            if num_steps > max_steps:
                # Give up
                break

            _, item = q.get()

            if item.length > self.dec_maxlen:
                # Discard item that's too long
                continue

            if (item.phoneme_id == self.eos_idx) and (item.parent is not None):
                pred_phonemes = []
                cur_item: typing.Optional[BeamSearchItem] = item.parent
                cur_score = 0.0
                while cur_item is not None:
                    if cur_item.phoneme_id not in {
                        self.sos_idx,
                        self.eos_idx,
                        self.pad_idx,
                    }:
                        pred_phonemes.append(self.phonemes[cur_item.phoneme_id])

                    cur_score += cur_item.score
                    cur_item = cur_item.parent

                if pred_phonemes:
                    completed_items.append((cur_score, list(reversed(pred_phonemes))))

                if len(completed_items) >= num_guesses:
                    break

                continue

            # Single step in decoder
            h = _grucell(
                item.dec,
                item.h,
                self.dec_w_ih,
                self.dec_w_hh,
                self.dec_b_ih,
                self.dec_b_hh,
            )  # (b, h)

            logits = np.matmul(h, self.fc_w.T) + self.fc_b

            # Get top indices
            top_pred = logits.argsort()[:, ::-1][0][:beam_width]
            top_prob = logits[:, top_pred][0]

            for pred, prob in zip(top_pred, top_prob):
                dec = typing.cast(np.ndarray, np.take(self.dec_emb, [pred], axis=0))
                new_item = BeamSearchItem(
                    dec=dec,
                    h=h,
                    phoneme_id=pred,
                    log_prob=prob + item.log_prob,
                    length=item.length + 1,
                    parent=item,
                )

                q.put((-new_item.score, new_item))

        return [
            result[1]
            for result in sorted(
                completed_items, key=lambda item: -item[0], reverse=True
            )
        ]
