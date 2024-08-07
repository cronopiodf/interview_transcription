### ¿Para qué sirve este proyecto?

Combina Whisper de Open IA y PyAnnote con la finalidad de hacer transcripciones y diarizaciones (identificación de hablantes) de entrevistas en procesos de investigación cualitativa. **Trabaja con entorno de ejecución GPU en Google Colab.**

Genera los siguientes archivos en la carpeta "audio_transcripción": 
1. archivo de audio way (audio.wav) #necesario para que PyAnnote trabaje
2. archivo trancripcion.json 
3. archivo transcripción_subtitulo.srt (se puede subir mientras se reproduce el audio para verificar transcripción)
4. Transcripción_pura.txt (toda la trancripción que hace whisper)
5. Identificación_hablantes.txt (cominación de la trancripción con la diarización)


### Requiere generar token en hugging-face

- Aceptar preferencias: https://hf.co/pyannote/segmentation-3.0
- Aceptar preferencias: https://huggingface.co/pyannote/speaker-diarization-3.1

- Generar token en: https://huggingface.co/settings/tokens

### En entorno Google Colab Instalar lo siguiente (https://colab.research.google.com)

```python
pip install pyannote.audio
```
```python
pip install --upgrade pyannote.audio
```
```python
pip install git+https://github.com/openai/whisper.git
```
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
```
```python
pip install ffmpeg
```
```python
!pip install pydub
```
```python
pip install setuptools-rust
```
```python
!pip install torch
```
```python
pip install torchaudio
```
```python
!pip install soundfile
```
```python
import locale

locale.getpreferredencoding = lambda: 'UTF-8'

!pip install librosa
```
#### FINALMENTE CLONAR DESDE GIT-HUB y ejecutar
```python
!git clone https://github.com/cronopiodf/interview_transcription.git
```
```python
%cd interview_transcription
```
```python
from interview_transcription import transcripcion_entrevistas
```

### Autoría

Este proyecto ha sido generado por David Añazco @cronopiodf bajo el proyecto de investigación: *Sustainable Educational Leadership in Ecuador: Research-Practice Partnerships for Leadership Policy and Practices in Secondary  Schools* que forma parte de su proceso de investigación doctoral PhD en la Universidad Católica de Lovaina.

##### Nota
Los archivos que genera este proyecto no son definitivos, se pueden entender como borradores para un proceso de pulida final de la trancripción y posterior análisis en proyectos de investigación.
