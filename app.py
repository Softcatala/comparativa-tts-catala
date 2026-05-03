import tempfile
import gradio as gr
import os
from TTS.utils.synthesizer import Synthesizer
from espeak_phonemizer import Phonemizer
from engine import Piper
from festival import festival_synthesize
from mms import MMS

MAX_TXT_LEN = 325

fonemitzador = Phonemizer("ca")

def carrega_bsc():
    model_path = os.getcwd() + "/models/bsc/best_model.pth"
    config_path = os.getcwd() + "/models/bsc/config.json"
    speakers_file_path = os.getcwd() + "/models/bsc/speakers.pth"
    vocoder_path = None
    vocoder_config_path = None

    synthesizer = Synthesizer(
        model_path, config_path, speakers_file_path, None, vocoder_path, vocoder_config_path,
    )

    return synthesizer

def carrega_collectivat():
    model_path = os.getcwd() + "/models/collectivat/fast-speech_best_model.pth"
    config_path = os.getcwd() + "/models/collectivat/fast-speech_config.json"
    vocoder_path = os.getcwd() + "/models/collectivat/ljspeech--hifigan_v2_model_file.pth"
    vocoder_config_path = os.getcwd() + "/models/collectivat/ljspeech--hifigan_v2_config.json"
    synthesizer = Synthesizer(
        model_path, config_path, None, None, vocoder_path, vocoder_config_path
    )

    return synthesizer

def carrega_piper():
    return Piper(os.getcwd() + "/models/piper/ca-upc_ona-x-low.onnx")

def carrega_mms():
    return MMS(os.getcwd() + "/models/mms")


model_bsc = carrega_bsc()
SPEAKERS = model_bsc.tts_model.speaker_manager.speaker_names

model_collectivat = carrega_collectivat()

model_piper = carrega_piper()

model_mms = carrega_mms()

request_count = 0

def tts(text, festival_voice, speaker_idx):
    if len(text) > MAX_TXT_LEN:
        text = text[:MAX_TXT_LEN]
        print(f"Input text was cutoff since it went over the {MAX_TXT_LEN} character limit.")
    print(text)

    # synthesize
    wav_bsc = model_bsc.tts(text, speaker_idx)
    wav_coll = model_collectivat.tts(text)
    wav_piper = model_piper.synthesize(text)

    fp_bsc = ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        model_bsc.save_wav(wav_bsc, fp)
        fp_bsc = fp.name

    fp_coll = ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        model_collectivat.save_wav(wav_coll, fp)
        fp_coll = fp.name

    fp_piper = ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        fp.write(wav_piper)
        fp_piper = fp.name
    
    fp_mms = ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        model_mms.synthesize(fp.name, text)
        fp_mms = fp.name

    fonemes = fonemitzador.phonemize(text, keep_clause_breakers=True)

    fp_festival = festival_synthesize(text, festival_voice)

    global request_count
    request_count += 1
    print(f"Requests: {request_count}")
    return fonemes, fp_festival, fp_bsc, fp_coll, fp_piper, fp_mms


description="""
Amb aquesta aplicació podeu sintetitzar text a veu amb alguns models neuronals lliures pel català i amb el motor Festival.

1. Model multi-parlant VITS entrenat pel BSC (Projecte Aina) [enllaç](https://huggingface.co/projecte-aina/tts-ca-coqui-vits-multispeaker)
2. Model Fastspeech entrenat per Col·lectivat [enllaç](https://github.com/CollectivaT-dev/TTS-API)
3. Model VITS entrenat per Piper/Home Assistant [enllaç](https://github.com/rhasspy/piper)
3. Model VITS entrenat per Meta (llicència CC-BY-NC) [enllaç](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)

El primer model ha estat entrenat amb totes les veus de FestCAT, els talls de Common Voice 8 i un altre corpus pel que conté moltes veus de qualitat variable. La veu d'Ona està seleccionada per defecte per la comparativa però podeu provar les altres.
Els models 2 i 3 han estat entrenats amb la veu d'Ona de FestCAT.
El model 4, anomenat MMS, de Meta (Facebook) ha estat entrenat a partir de dades d'un [audiollibre](http://live.bible.is/bible/CATBSS/LUK/1) de la Bíblia

Aquesta aplicació fa servir l'últim estat de l'espeak millorat per Carme Armentano del BSC
https://github.com/projecte-aina/espeak-ng

NOTA: El model de col·lectivat treballa amb grafemes pel que no fa servir espeak com a fonemitzador. Festival conté les seves pròpies normes fonètiques.

ACTUALITZACIÖ (Agost 2024): El BSC en el marc del projecte Aina ha publicat nous models, podeu provar-los [aquí](https://huggingface.co/spaces/projecte-aina/matxa-alvocat-tts-ca).
"""
article= ""

iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.Textbox(
            label="Text",
            value="L'Èlia i l'Alí a l'aula.  L'oli i l'ou.  Lulú olorava la lila.",
        ),
        gr.Dropdown(label="Parlant del motor Festival", choices=["ona", "pau"], value="ona"),
        gr.Dropdown(label="Parlant del model VITS multi-parlant del BSC", choices=SPEAKERS, value="ona")
    ],
    outputs=[
        gr.Markdown(label="Fonemes"),
        gr.Audio(label="Festival",type="filepath"),
        gr.Audio(label="BSC VITS",type="filepath"),
        gr.Audio(label="Collectivat Fastspeech",type="filepath"),
        gr.Audio(label="Piper VITS",type="filepath"),
        gr.Audio(label="Meta MMS VITS",type="filepath")
    ],
    title="Comparativa de síntesi lliure en català️",
    description=description,
    article=article,
    allow_flagging="never",
    layout="vertical",
    live=False,
    examples=[
        ["Duc pà sec al sac, m'assec on sóc i el suco amb suc", "ona", "ona"],
        ["Un plat pla blanc, ple de pebre negre n’era. Un plat blanc pla, ple de pebre negre està", "ona", "ona"],
        ["Visc al bosc i busco vesc i visc del vesc que busco al bosc", "ona", "ona"],
        ["Una polla xica, pica, pellarica, camatorta i becarica va tenir sis polls xics, pics, pellarics, camacurts i becarics. Si la polla no hagués sigut xica, pica, pellarica, camatorta i becarica, els sis polls no haurien sigut xics, pics, pellarics, camacurts i becarics.", "ona", "ona"]
    ]
)
iface.launch(server_name="0.0.0.0", server_port=7860)
