import logging
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    Form, 
    HTTPException,
    status
)
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 

from handler import EndpointHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="dlgen-AI-project",
    description="Audio Genre Classification API using AST and Wav2Vec2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Starting up: Loading ensemble models into memory...")
model_handler = EndpointHandler(path=".") 
logger.info("Models loaded successfully! API is ready.")



@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("batch"),
    temperature: float = Form(1.0),
    verbose: bool = Form(False)
):
    try:
        audio_bytes = await file.read()
        
        # Format the data exactly how your handler's `_parse_request` expects it
        hf_formatted_payload = {
            "inputs": audio_bytes,
            "parameters": {
                "mode": mode,
                "temperature": temperature,
                "verbose": verbose
            }
        }
        
        # Pass the payload to your handler's __call__ method
        result = model_handler(hf_formatted_payload)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Model handler error: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=result["error"]
            )

            
        return result
        
    except Exception as e:
        logger.error(f"FastAPI route error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  
            detail=str(e)
        )
