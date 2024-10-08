from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile, TemporaryDirectory
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
import httpx
import PyPDF2
import google.generativeai as genai
import spacy
from pymongo import MongoClient

app = FastAPI()

load_dotenv()

key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-1.5-flash')

client = MongoClient(os.getenv('mongoURL'), username= os.getenv('username'), password = os.getenv('password'))  # Adjust the URI accordingly
db = client['admin']
mycol = db['ResumeSummary']

def insert(ResumeName, Company, Position, YOE, Skills, Experiences, Projects, Awards):
    myquery = {'ResumeName': ResumeName,
        'Company': Company,
        'Position': Position,
        'Years of Experience': YOE, 
        "Skills": Skills, 
        "Experiences": Experiences, 
        "Projects": Projects, 
        "Awards": Awards}

    cursor = mycol.find(myquery)

    if(len(list(cursor)) == 0): mycol.insert_one(myquery)


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from the PDF: {e}")
        return None


def summarize_resume(text):
    try:
        response = model.generate_content(
            'The following text is an applicant\'s resume. You must sort it and summarize it into the following categories: years of experience, skills, experiences, projects, awards. Your output should have a very brief format, with the total years of experience listed on a new line after "Years of Experience." The list of skills should be listed on a new line separate from "Skills: " The list of experiences with brief descriptions should be listed on a new line after "Experiences." The list of projects with brief descriptions should be listed on a new line after "Projects." A list of the awards should be listed on a new line "Awards." Nothing else should be output. Don\'t give any introduction before giving this output, all I need is this output and nothing else. You must include all the fields I mentioned and label them as I specified. If you cannot find information for any of these fields, label it n/a. Here is the resume:' + text,
        )
        return response.text
    except httpx.RequestError as e:
        print(f"An HTTP request error occurred: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"An HTTP status error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while summarizing the resume: {e}")
        return None

def parse_summary_with_spacy(summary_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(summary_text)
    Years_of_Experience = []
    skills = []
    experiences = []
    projects = []
    awards = []

    result = {}

    current_category = None

    for line in summary_text.split('\n'):
        line = line.strip()
        if line.startswith("Years of Experience"):
            current_category = "Years of Experience"
        elif line.startswith("Skills"):
            current_category = "Skills"
        elif line.startswith("Experiences"):
            current_category = "Experiences"
        elif line.startswith("Projects"):
            current_category = "Projects"
        elif line.startswith("Awards"):
            current_category = "Awards"

        try:
            line = line.split(":")[1].strip()
        except:
            line = line

        if current_category == "Years of Experience":
            Years_of_Experience.append(line)
        elif current_category == "Skills":
            skills.append(line)
        elif current_category == "Experiences":
            experiences.append(line)
        elif current_category == "Projects":
            projects.append(line)
        elif current_category == "Awards":
            awards.append(line)

    return {
        "Years of Experience": " ".join(Years_of_Experience),
        "Skills": " ".join(skills),
        "Experiences": " ".join(experiences),
        "Projects": " ".join(projects),
        "Awards": " ".join(awards)
    }


session = boto3.session.Session()
client = session.client('s3',
                        endpoint_url=os.getenv('endpoint'),
                        region_name='nyc3',
                        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'))


@app.post("/summarize/")
async def summarize(filekey: str):
    with TemporaryDirectory() as temp_dir:
        file_location = os.path.join(temp_dir, filekey)
        try:
            client.download_file(os.getenv('Bucket1'), filekey, file_location)
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise HTTPException(status_code=403, detail="Credentials error")

        extracted = extract_text_from_pdf(file_location)
        if extracted is None:
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

        insights = summarize_resume(extracted)
        if insights is None:
            raise HTTPException(status_code=500, detail="Failed to summarize resume")

        parse_data = parse_summary_with_spacy(insights)
        if parse_data is None:
            raise HTTPException(status_code=500, detail="Failed to parse resume")
        
        insert(filekey, " ", " ", parse_data['Years of Experience'], parse_data["Skills"], parse_data["Experiences"], parse_data['Projects'], parse_data['Awards'])

        with NamedTemporaryFile(delete=True) as temp_file:
            for key, value in parse_data.items():
                temp_file.write(f'{key}:{value}\n'.encode('utf-8'))
            temp_file.flush()  # Ensure data is written

            client.upload_file(temp_file.name, os.getenv('Bucket2'), filekey + ' Summary')

        return JSONResponse(content={"transcription": "Summary Uploaded Successfully"})


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
