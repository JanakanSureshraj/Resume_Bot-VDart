from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisperx
import torch
import os

app = FastAPI()

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# Load the WhisperX model
model = whisperx.load_model("base", DEVICE, compute_type=compute_type)


if not os.path.exists('temp'):
    os.makedirs('temp')


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Transcribe the audio file using WhisperX
        result = model.transcribe(file_location)
        transcription = result['segments'][0]['text']

        # Remove the temporary file
        os.remove(file_location)
        
        #returns transcription
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"