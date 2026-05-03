#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) 2016 Jordi Mas i Hernandez <jmas@softcatala.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.

import subprocess
import tempfile

festival_voices = {
    "ona": "voice_upc_ca_ona_hts",
    "pau": "voice_upc_ca_pau_hts"
}

def _normalize(result):
    mapping = {
                '’' : '\'',
                'à' : 'à',
                'í' : 'í',
                'ó' : 'ó',
                'è' : 'è',
                'ò' : 'ò',
                'ú' : 'ú',
              }

    for char in mapping.keys():
        result = result.replace(char, mapping[char])

    return result


def festival_synthesize(text, voice):
    if voice not in ["ona", "pau"]:
        raise Error

    txt2wave = '/usr/bin/text2wave'

    with tempfile.NamedTemporaryFile() as encoded_file,\
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wave_file:

        text = _normalize(text)
        f = open(encoded_file.name, 'wb')
        f.write(text.encode('ISO-8859-15', 'ignore'))
        f.close()

        cmd = '{0} -o {1} {2} -eval "({3})"'.\
              format(txt2wave, wave_file.name, encoded_file.name, festival_voices[voice])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p.wait()

        return wave_file.name
