"""
Transcripción + diarización de entrevistas y focus groups (versión octubre 2026)

Herramientas:
  - WhisperX (faster-whisper + alineación por palabra con wav2vec2)
  - pyannote.audio 4 con el modelo "speaker-diarization-community-1"

Instalación (una vez):
  pip install whisperx          # instala también pyannote.audio y faster-whisper
  + ffmpeg instalado en el sistema
  + aceptar las condiciones de https://huggingface.co/pyannote/speaker-diarization-community-1
  + token de lectura de Hugging Face: https://huggingface.co/settings/tokens
"""

import gc
import json
import os

import torch
import whisperx
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd

# ---------------- Parámetros configurables ----------------
OUTPUT_DIR = "audio_transcripción"
MODELO_WHISPER = "large-v3"   # máxima precisión; "large-v3-turbo" es ~5x más rápido con algo menos de precisión
BATCH_SIZE = 16               # bájelo (8, 4) si la GPU se queda sin memoria
MODELO_DIARIZACION = "pyannote/speaker-diarization-community-1"
# -----------------------------------------------------------


def format_timestamp(seconds, sep=","):
    """Segundos -> hh:mm:ss,mmm (formato SRT)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def liberar():
    """Libera memoria de GPU entre etapas."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_and_diarize(audio_path, output_dir, num_speakers, hf_token, language=None):
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        print("⚠ GPU no disponible: se usará la CPU (funciona, pero mucho más lento).")
        device, compute_type = "cpu", "int8"

    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar audio (cualquier formato: mp3, wav, m4a, mp4...; ffmpeg convierte a 16 kHz mono)
    audio = whisperx.load_audio(audio_path)

    # 2. Transcripción
    print("1/3 Transcribiendo...")
    model = whisperx.load_model(MODELO_WHISPER, device, compute_type=compute_type, language=language)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    language = result["language"]
    del model
    liberar()

    # 3. Alineación palabra por palabra (marcas de tiempo precisas)
    print("2/3 Alineando palabras...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)
    del model_a
    liberar()

    # 4. Diarización con barra de progreso
    print("3/3 Identificando hablantes...")
    pipeline = Pipeline.from_pretrained(MODELO_DIARIZACION, token=hf_token)
    pipeline.to(torch.device(device))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    with ProgressHook() as hook:
        output = pipeline({"waveform": waveform, "sample_rate": 16000},
                          num_speakers=num_speakers, hook=hook)
    # Diarización "exclusiva" (sin solapamientos): la recomendada para cruzar con la transcripción
    diarization = getattr(output, "exclusive_speaker_diarization", output)
    diarize_df = pd.DataFrame(
        [{"start": t.start, "end": t.end, "speaker": spk}
         for t, _, spk in diarization.itertracks(yield_label=True)]
    )
    # Asigna hablante a cada palabra; fill_nearest evita que se pierda texto sin hablante
    result = whisperx.assign_word_speakers(diarize_df, result, fill_nearest=True)
    result["language"] = language
    segments = result["segments"]

    # 5. Archivos de salida
    # 5a. JSON completo (segmentos, palabras, tiempos, hablantes)
    with open(os.path.join(output_dir, "transcripcion.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 5b. Texto plano
    with open(os.path.join(output_dir, "transcripción_pura.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(s["text"].strip() for s in segments))

    # 5c. Subtítulos SRT (con hablante)
    with open(os.path.join(output_dir, "transcripcion_subtitulo.srt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, start=1):
            spk = s.get("speaker", "DESCONOCIDO")
            f.write(f"{i}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n"
                    f"[{spk}] {s['text'].strip()}\n\n")

    # 5d. Transcripción por hablante (turnos consecutivos fusionados, con hora de inicio)
    turnos = []
    for s in segments:
        spk = s.get("speaker", "DESCONOCIDO")
        if turnos and turnos[-1]["speaker"] == spk:
            turnos[-1]["text"] += " " + s["text"].strip()
        else:
            turnos.append({"speaker": spk, "start": s["start"], "text": s["text"].strip()})
    with open(os.path.join(output_dir, "identificación_hablantes.txt"), "w", encoding="utf-8") as f:
        for t in turnos:
            f.write(f"[{format_timestamp(t['start'], '.')[:8]}] {t['speaker']}: {t['text']}\n\n")

    print(f"Listo. Archivos guardados en '{output_dir}' (idioma detectado: {language}).")


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN") or input(
        "Token de Hugging Face (acepte antes las condiciones de "
        "https://huggingface.co/pyannote/speaker-diarization-community-1): ").strip()
    audio_path = input("Ruta del archivo de audio/vídeo (clic derecho > copiar ruta y pegar): ").strip().strip('"\'')
    n = input("Número de hablantes (Enter = detección automática): ").strip()
    lang = input("Código de idioma, p. ej. 'es' (Enter = detección automática): ").strip() or None

    transcribe_and_diarize(audio_path, OUTPUT_DIR, int(n) if n else None, hf_token, lang)
