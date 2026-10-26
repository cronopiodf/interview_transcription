(english below)

## ¿Para qué sirve este proyecto?

Transcribe entrevistas y focus groups e identifica quién habla en cada momento (diarización), para procesos de investigación cualitativa. Combina **WhisperX** (transcripción con tiempos precisos palabra por palabra) y **pyannote community-1** (identificación de hablantes). **Funciona en Google Colab con GPU**, sin instalar nada en tu computadora.

Acepta audio o vídeo en casi cualquier formato: mp3, wav, m4a, mp4…

Genera los siguientes archivos en la carpeta `audio_transcripción`:

1. `transcripcion.json`: todos los datos (frases, palabras, tiempos y hablantes). Útil para análisis posteriores.
2. `transcripción_pura.txt`: el texto completo, sin hablantes ni tiempos.
3. `transcripcion_subtitulo.srt`: subtítulos con el hablante de cada frase. Puedes cargarlos en un reproductor (por ejemplo VLC) mientras escuchas el audio para revisar la transcripción. Modifica visualización y carga el subtitulo. Lo verás así: 
  <img width="600" height="300" alt="imagen" src="https://github.com/user-attachments/assets/87f7e2d1-46dd-454a-a314-c6bef0d5f8fd" />

5. `identificación_hablantes.txt`: la transcripción organizada por turnos de palabra, con la hora de inicio de cada turno: 

```
[00:02:15] SPEAKER_00: ¿Cómo describiría el liderazgo en su institución?

[00:02:21] SPEAKER_01: Bueno, yo diría que es compartido...
```
6. `identificación_hablantes.rtf`: la misma transcripción por turnos, con formato (título, hablantes en negrita y tiempos en gris). Se abre directamente en Word o LibreOffice y se puede importar en NVivo o Atlas.ti. Es el archivo más cómodo para revisar, sustituir los hablantes por seudónimos y empezar a codificar. Por ejemplo:
   <img width="600" height="200" alt="imagen" src="https://github.com/user-attachments/assets/55071ab7-d372-4fd7-ba15-b8de3f9ca045" />


---

## Paso 1. Preparar tu cuenta de Hugging Face (solo la primera vez)

1. Crea una cuenta gratuita en https://huggingface.co
2. Entra en https://huggingface.co/pyannote/speaker-diarization-community-1 y acepta las condiciones de uso.
3. Crea un token de lectura ("Read") en https://huggingface.co/settings/tokens y guárdalo. Es una clave que el programa te pedirá más adelante. NOTA: No compartas ni piedas esta clave, consérvala en un lugar seguro. 

> Si ya tenías un token de la versión anterior, puedes reutilizarlo, pero **debes aceptar igualmente las condiciones del paso 2**: el modelo es nuevo.

## Paso 2. Abrir Google Colab con GPU

1. Entra en https://colab.research.google.com y crea un cuaderno nuevo.
2. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4** → Guardar.

## Paso 3. Subir el audio

En el panel izquierdo, haz clic en el icono de la carpeta 📁 y arrastra tu archivo de audio. Cuando termine de subirse, haz clic derecho sobre el archivo → **Copiar ruta**.

## Paso 4. Instalar y ejecutar

Copia cada bloque en una celda y ejecútalo (botón ▶) en orden:

```python
!pip install whisperx
```

> Si Colab pide **reiniciar la sesión** al terminar, acepta y continúa con la celda siguiente.

```python
!git clone https://github.com/cronopiodf/interview_transcription.git
%cd interview_transcription
```

```python
%run diarizacion.py
```

El programa te hará cuatro preguntas:

1. **Token de Hugging Face**: pega el token del Paso 1.
2. **Ruta del archivo**: pega la ruta copiada en el Paso 3.
3. **Número de hablantes**: escribe el número (por ejemplo `2` para una entrevista). Si no lo sabes, pulsa Enter y se detectará automáticamente.
4. **Idioma**: escribe `es` para español (recomendado: mejora la precisión) o pulsa Enter para detección automática.

Verás el avance en tres etapas: transcripción, alineación e identificación de hablantes. Una entrevista de una hora suele tardar pocos minutos.

## Paso 5. Descargar los resultados

En el panel de archivos abre `interview_transcription/audio_transcripción`, haz clic derecho sobre cada archivo → **Descargar**. Los archivos de Colab se borran al cerrar la sesión, así que descárgalos antes.

---

## Problemas frecuentes

- **"CUDA out of memory"**: abre `diarizacion.py`, cambia `BATCH_SIZE = 16` por `8` o `4` y vuelve a ejecutar.
- **Error 401 / 403 o "gated repo"**: no se aceptaron las condiciones del modelo (Paso 1.2) o el token es incorrecto.
- **"GPU no disponible"**: revisa el Paso 2. El programa sigue funcionando en CPU, pero mucho más lento.
- **¿Quieres más velocidad?** En `diarizacion.py`, cambia `MODELO_WHISPER = "large-v3"` por `"large-v3-turbo"` (mucho más rápido, algo menos preciso).

## Nota sobre ética y privacidad

Al usar Google Colab, el audio se sube temporalmente a servidores de Google. Verifica que esto sea compatible con el consentimiento informado y el protocolo ético de tu investigación. Si no lo es, el script también puede ejecutarse en una computadora propia con GPU NVIDIA.

## Nota

Los archivos que genera este proyecto no son definitivos: son borradores para una revisión final de la transcripción y su posterior análisis. La identificación de hablantes puede fallar con voces superpuestas o intervenciones muy breves; revísala siempre escuchando el audio. Los hablantes aparecen como `SPEAKER_00`, `SPEAKER_01`…; sustitúyelos por los seudónimos de tu investigación.

## Autoría

Este proyecto ha sido generado por David Añazco @cronopiodf con la asitencia de claude code. Este es un proyecto estudiantil gracias a la experencia desarrollada bajo el proyecto de investigación: *Sustainable Educational Leadership in Ecuador: Research-Practice Partnerships for Leadership Policy and Practices in Secondary Schools*, que forma parte de su proceso de investigación doctoral (PhD) en la Universidad Católica de Lovaina. Actualizado en octubre de 2026.

---
---

## What is this project for?

It transcribes interviews and focus groups and identifies who is speaking at each moment (speaker diarization) for qualitative research. It combines **WhisperX** (transcription with precise word-level timestamps) and **pyannote community-1** (speaker identification). **It runs on Google Colab with a GPU**, with nothing to install on your own computer.

It accepts audio or video in almost any format: mp3, wav, m4a, mp4…

The following files are generated in the `audio_transcripción` folder:

1. `transcripcion.json`: all the data (sentences, words, timestamps and speakers). Useful for further analysis.
2. `transcripción_pura.txt`: the full text, without speakers or timestamps.
3. `transcripcion_subtitulo.srt`: subtitles with the speaker of each sentence. Load them in a player (e.g. VLC) while listening to the audio to check the transcription. You must select a mode of visualization, upload subtitle file (.srt) and reproduce. You will see something like this:
   <img width="600" height="300" alt="imagen" src="https://github.com/user-attachments/assets/87f7e2d1-46dd-454a-a314-c6bef0d5f8fd" />
   
5. `identificación_hablantes.txt`: the transcription organized by speaking turns, with the start time of each turn.
6. 5. `identificación_hablantes.rtf`: the same turn-by-turn transcription, formatted (title, speakers in bold, timestamps in grey). It opens directly in Word or LibreOffice and can be imported into NVivo or Atlas.ti. It is the most convenient file for reviewing, replacing speakers with pseudonyms and starting to code. For instance:
      <img width="1119" height="505" alt="imagen" src="https://github.com/user-attachments/assets/55071ab7-d372-4fd7-ba15-b8de3f9ca045" />
      

## Step 1. Set up your Hugging Face account (first time only)

1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the user conditions.
3. Create a "Read" token at https://huggingface.co/settings/tokens and keep it. The program will ask for it.

> If you already had a token from the previous version you can reuse it, but **you must still accept the conditions in step 2**: the model is new.

## Step 2. Open Google Colab with a GPU

1. Go to https://colab.research.google.com and create a new notebook.
2. Menu **Runtime → Change runtime type → T4 GPU** → Save.

## Step 3. Upload the audio

In the left panel, click the folder icon 📁 and drag in your audio file. Once uploaded, right-click the file → **Copy path**.

## Step 4. Install and run

Paste each block into a cell and run it (▶ button) in order:

```python
!pip install whisperx
```

> If Colab asks you to **restart the session** when it finishes, accept and continue with the next cell.

```python
!git clone https://github.com/cronopiodf/interview_transcription.git
%cd interview_transcription
```

```python
%run diarizacion.py
```

The program will ask four questions:

1. **Hugging Face token**: paste the token from Step 1.
2. **File path**: paste the path copied in Step 3.
3. **Number of speakers**: type the number (e.g. `2` for an interview). If unknown, press Enter for automatic detection.
4. **Language**: type a language code such as `es` or `en` (recommended: improves accuracy), or press Enter for automatic detection.

You will see progress in three stages: transcription, alignment and speaker identification. A one-hour interview usually takes a few minutes.

## Step 5. Download the results

In the file panel open `interview_transcription/audio_transcripción`, right-click each file → **Download**. Colab files are deleted when the session ends, so download them first.

## Common problems

- **"CUDA out of memory"**: open `diarizacion.py`, change `BATCH_SIZE = 16` to `8` or `4` and run again.
- **Error 401 / 403 or "gated repo"**: the model conditions were not accepted (Step 1.2) or the token is wrong.
- **"GPU no disponible"** (GPU not available): check Step 2. The program still works on CPU, but much more slowly.
- **Need more speed?** In `diarizacion.py`, change `MODELO_WHISPER = "large-v3"` to `"large-v3-turbo"` (much faster, slightly less accurate).

## Ethics and privacy note

When using Google Colab, the audio is temporarily uploaded to Google's servers. Check that this is compatible with your participants' informed consent and your research ethics protocol. If it is not, the script can also run on your own computer with an NVIDIA GPU.

## Note

The files generated by this project are not final: they are drafts for a final review of the transcription and subsequent analysis. Speaker identification may fail with overlapping voices or very short interventions; always check it against the audio. Speakers appear as `SPEAKER_00`, `SPEAKER_01`…; replace them with your study's pseudonyms.

## Authorship

This project was created by David Añazco @cronopiodf with the assitance of claude code. This is a student tool as part of the research project *Sustainable Educational Leadership in Ecuador: Research-Practice Partnerships for Leadership Policy and Practices in Secondary Schools*, which is part of his PhD research at the Catholic University of Louvain. Updated October 2026.
