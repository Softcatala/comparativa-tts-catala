# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import commons
import utils
from models import SynthesizerTrn
from scipy.io.wavfile import write
from pathlib import Path
from typing import Union

class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file).readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt

class MMS():
    def __init__(self, model_path: Union[str, Path]):
        ckpt_dir = model_path
        vocab_file = f"{ckpt_dir}/vocab.txt"
        config_file = f"{ckpt_dir}/config.json"
        assert os.path.isfile(config_file), f"{config_file} doesn't exist"
        self.hps = utils.get_hparams_from_file(config_file)
        self.text_mapper = TextMapper(vocab_file)
        self.net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        g_pth = f"{ckpt_dir}/G_100000.pth"
        print(f"load {g_pth}")

        _ = utils.load_checkpoint(g_pth, self.net_g, None)

    def synthesize(self, wav_path: str, txt):
        print(f"text: {txt}")
        txt = txt.lower()
        txt = self.text_mapper.filter_oov(txt)
        stn_tst = self.text_mapper.get_text(txt, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            hyp = self.net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0,0].cpu().float().numpy()

        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        print(f"wav: {wav_path}")
        write(wav_path, self.hps.data.sampling_rate, hyp)
        return