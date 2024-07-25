import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import torchaudio
import os
import json
from pydub import AudioSegment

def format_timestamp(seconds):
    """Convierte el tiempo en segundos al formato de tiempo SRT (hh:mm:ss,mss)"""
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(transcription, srt_path):
    """Genera un archivo SRT a partir de la transcripción proporcionada"""
    with open(srt_path, "w", encoding="utf-8") as file:
        for i, segment in enumerate(transcription["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

def transcribe_and_diarize(audio_file_path, output_dir, num_speakers, auth_token):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU no disponible. Este script requiere una GPU para ejecutarse.")

    os.makedirs(output_dir, exist_ok=True)

    try:
        audio = AudioSegment.from_mp3(audio_file_path)
        wav_audio_file_path = os.path.join(output_dir, "audio.wav")
        audio.export(wav_audio_file_path, format="wav")
    except Exception as e:
        print(f"Error al convertir el archivo de audio: {e}")
        return

    try:
        model = whisper.load_model("medium", device="cuda")
        result = model.transcribe(wav_audio_file_path, fp16=True, verbose=False)
    except Exception as e:
        print(f"Error al transcribir el archivo de audio: {e}")
        return

    transcription_output_path = os.path.join(output_dir, "transcripcion.json")
    with open(transcription_output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    complete_transcription_path = os.path.join(output_dir, "transcripción_pura.txt")
    with open(complete_transcription_path, "w", encoding="utf-8") as file:
        file.write(result["text"])

    srt_path = os.path.join(output_dir, "transcripcion_subtitulo.srt")
    generate_srt(result, srt_path)

    try:
        waveform, sample_rate = torchaudio.load(wav_audio_file_path)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.31",
            use_auth_token=auth_token
        )
        pipeline.to(torch.device("cuda"))

        with ProgressHook() as hook:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers, hook=hook)
    except Exception as e:
        print(f"Error en la diarización de hablantes: {e}")
        return

    speaker_transcription = []

    current_speaker = None
    current_text = ""
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        speaker = None
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if turn.start <= start_time < turn.end:
                speaker = speaker_label
                break

        if speaker is not None:
            if speaker == current_speaker:
                current_text += text + " "
            else:
                if current_speaker is not None:
                    speaker_transcription.append((current_speaker, current_text.strip(), start_time))
                current_speaker = speaker
                current_text = text + " "

    if current_speaker is not None:
        speaker_transcription.append((current_speaker, current_text.strip(), start_time))

    output_file_path = os.path.join(output_dir, "identificación_hablantes.txt")
    with open(output_file_path, "w", encoding="utf-8") as file:
        for speaker, text, start_time in speaker_transcription:
            file.write(f"speaker_{speaker}: {text}\n")

    print(f"Transcripción, diarización de hablantes y archivo SRT completadas y guardadas en el directorio '{output_dir}'.")

# Solicitar al usuario los parámetros necesarios
auth_token = input("Ingrese su token de Pyannote, siga las indicaciones según https://github.com/pyannote/pyannote-audio: ")
audio_file_path = input("Ingrese la dirección del archivo de audio en formato MP3, click derecho sobre el archivo, copiar ruta y pegar: ")
num_speakers = int(input("Ingrese el número de hablantes: "))

# Parámetros configurables
output_dir = "audio_transcripción"

transcribe_and_diarize(audio_file_path, output_dir, num_speakers, auth_token)