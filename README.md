Qué hace este script
Combina Whisper de Open IA y PyAnnote para hacer transcripciones y diarizaciones (identificación de hablantes). Trabaja con entorno de ejecución GPU en Google Colab. 


Requiere 

Aceptar preferencias: https://hf.co/pyannote/segmentation-3.0

Aceptar preferencias: https://huggingface.co/pyannote/speaker-diarization-3.1

Generar token en: https://huggingface.co/settings/tokens 



En entorno de https://colab.research.google.com/ instalar lo siguiente:



pip install pyannote.audio

pip install --upgrade pyannote.audio

pip install git+https://github.com/openai/whisper.git

pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

pip install ffmpeg

!pip install pydub

pip install setuptools-rust

!pip install torch

pip install torchaudio

!pip install soundfile




import locale

locale.getpreferredencoding = lambda: 'UTF-8'

!pip install librosa	


FINALMENTE CLONAR REPOSITORIO Y EJECUTAR

!git clone https://github.com/cronopiodf/interview_transcription.git


%cd interview_transcription


from interview_transcription import transcripcion_entrevistas
