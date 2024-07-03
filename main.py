from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from fastapi.middleware.cors import CORSMiddleware
import os
# uvicorn main:app --reload
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\vedan\\OneDrive\Documents\\GeminiDynamo\\mission-dynamoCards-v2\\authentication.json'
os.environ['GRPC_DNS_RESOLVER'] = 'native'

class ConceptDefinition(BaseModel):
    concepts: Dict[str, str] = Field(description="A dictionary of key concepts and their definitions")
    #concept: str = Field(description="A key concept found in the text")
    #definition: str = Field(description="Definition of the key concept")
# Set up the parser with the Pydantic model
parser = JsonOutputParser(pydantic_object=ConceptDefinition)

from services.genai import (
    YoutubeProcessor,
    GeminiProcessor
)

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl
    # advanced settings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai_processor = GeminiProcessor(
        model_name = "gemini-pro",
        project = "gemini-quizzify-427910" #"gemini-dynamo-428115" #ai-dev-cqc-q1-2024 #gemini-quizzify-427910
    )

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    # Doing the analysis
    processor = YoutubeProcessor(genai_processor = genai_processor,parser=parser)
    result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose=False)
    
    #summary = genai_processor.generate_document_summary(result, verbose=True)
    
    # Find key concepts
    raw_concepts = processor.find_key_concepts(result, verbose=True)
    
    # Deconstruct
    #unique_concepts = {}
    #for concept_dict in raw_concepts:
    #    print(concept_dict)
    #    for key, value in concept_dict.items():
    #        unique_concepts[key] = value
    
    # Reconstruct
    #key_concepts_list = [{key: value} for key, value in concept_dict.items()]
    key_concepts_list = []
    for content in raw_concepts:
        key_concepts_list.extend([{'term':key,'definition':value} for key, value in content['concepts'].items()])
    #key_concepts_list = [{key: value} for key, value in raw_concepts[i]['concepts'].items() for i in range(len(raw_concepts))]
    return {
        "key_concepts": key_concepts_list
    }
