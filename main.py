from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisperx
import torch
from typing import List
import magic
from pydub import AudioSegment
import os

# Checking if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# Load the Whisper model:
model = whisperx.load_model("base", DEVICE, compute_type=compute_type)
app = FastAPI()

# Check file format and convert to WAV if needed
async def convert_to_wav(file: UploadFile) -> str:
    # Determine MIME type of the uploaded file
    mime = magic.Magic(mime=True)
    file_mime = mime.from_buffer(await file.read())
    
    # Check if file is not in WAV format, then convert
    if not file_mime.startswith('audio/wav'):
        # Create a temporary file.
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            # Write the user's uploaded file to the temporary file.
            temp_wav.write(await file.read())
            return temp_wav.name
    else:
        return file.filename

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    # For each file, let's store the results in a list of dictionaries.
    results = []

    for file in files:
        # Convert to WAV if needed
        wav_file_path = await convert_to_wav_if_needed(file)
        
        # Let's get the transcript of the temporary file.
        result = model.transcribe(wav_file_path)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"],
                                          device=DEVICE)    

        result = whisperx.align(result["segments"], model_a,
                                metadata,
                                wav_file_path,
                                DEVICE,
                                return_char_alignments=False)

        # Now we can store the result object for this file.
        results.append({
            'filename': file.filename,
            'transcript': result['segments'],
        })

        # Delete temporary WAV file if conversion was needed
        if not file.filename.endswith('.wav'):
            os.remove(wav_file_path)
                                              
    return JSONResponse(content={'results': results})

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
