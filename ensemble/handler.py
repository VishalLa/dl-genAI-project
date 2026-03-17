import os 
import json 
import tempfile 
import logging 

from pathlib import Path
from typing import Dict, Any

import torch
import torchaudio 
import torch.nn as nn 

logger = logging.getLogger(__name__)

class EndpointHandler: 

    GENRE_MAP = {
        "blues":     0,
        "classical": 1,
        "country":   2,
        "disco":     3,
        "hiphop":    4,
        "jazz":      5,
        "metal":     6,
        "pop":       7,
        "reggae":    8,
        "rock":      9,
    }

    def __init__(self, path: str = ""):

        self.repo_dir = Path(path) if path else Path(".")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading ensemble from {self.repo_dir} on {self.device} ...")

        import sys 
        sys.path.insert(0, str(self.repo_dir))
        from model_arch.ast_model import ASTGenreClassifier
        from ensemble.model_arch.wav2vec2_model import Wav2Vec2ForGenre 
        from prediction_pipeline import PredictionPipeline, load_ensemble

        self.ast_model_fn = lambda: ASTGenreClassifier(num_classes=len(self.GENRE_MAP))
        self.wav2vec2_model_fn = lambda: Wav2Vec2ForGenre(num_classes=len(self.GENRE_MAP))

        self.json_path = self.repo_dir / "modelpaths.json"

        models = load_ensemble(
            ast_model_fn = self.ast_model_fn,
            wav2vec2_model_fn=self.wav2vec2_model_fn, 
            json_path = self.json_path,
            device = self.device
        )

        self.pipeline = PredictionPipeline(
            models = models,
            genre_map = self.GENRE_MAP, 
            batch_size = 8, 
            ratio = None,
            temperature = 1.0   # can be overridden per request
        )

        n_ast = len(models["ast-models"])
        n_w2v = len(models["wav2vec2-models"])
        logger.info(f"Endpoint ready — {n_ast} AST + {n_w2v} Wav2Vec2 models loaded.")


    def _parse_request(self, data: Any):
        """
        Normalise the two supported input formats into (audio_bytes, params).

          - {"inputs": <bytes>,  "parameters": {...}}   ← binary via JSON wrapper
          - {"inputs": <base64>, "parameters": {...}}   ← base64 string
          - raw bytes directly                          ← Content-Type: audio/*
        """

        params = {}

        # Raw binary (Content-Type: audio/wav, audio/mp3, etc.)
        if isinstance(data, bytes):
            return data, params 
        
        # Dict payload — HF standard format
        if isinstance(data, dict):
            params = data.get("parameters", {})
            raw_input = data.get("inputs", b"")

            if isinstance(raw_input, bytes):
                return raw_input, params 

            if isinstance(raw_input, str):
                import base64 
                return base64.b64decode(raw_input), params 
            
        raise ValueError(
            f"Unsupported input type: {type(data)}. "
            "Send raw audio bytes or a JSON dict with 'inputs' key."
        )

    
    def __call__(self, data: Any) -> Dict[str, Any]:
        """
        Called on every inference request.

        Handles both raw binary audio and base64 JSON payloads.
        Writes audio to a temp file so torchaudio can load it.
        """
        
        try: 
            audio_bytes, params = self._parse_request(data=data)
            mode = params.get("mode", "batch")
            temperature = params.get("temperature", self.pipeline.temperature)
            verbose = params.get("verbose", False)

            original_temp = self.pipeline.temperature
            self.pipeline.temperature = temperature 

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp: 
                temp.write(audio_bytes)
                temp_path = Path(temp.name)

            try: 
                predict_fn = (
                    self.pipeline.predict_batch if mode == "batch" else self.pipeline.single_prediction
                )

                result = predict_fn(
                    audio_path=temp_path,
                    verbose=verbose,
                    return_json=True
                )
            finally: 
                temp_path.unlink(missing_ok=True)
                self.pipeline.temperature = original_temp

            return result


        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return {"error": str(e)}
        
