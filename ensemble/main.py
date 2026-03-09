from ast_model import ASTGenreClassifier 
from wav2vec2_model import Wav2Vec2ForGenre
from prediction_pipeline import PredictionPipeline, load_ensemble

import torch 
import pandas as pd
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 10

genre_map = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
    'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
}

ast_model_fn = lambda: ASTGenreClassifier(num_classes=num_classes)
wav2vec2_model_fn = lambda: Wav2Vec2ForGenre(num_classes=num_classes)

models = load_ensemble(
    ast_model_fn=ast_model_fn,
    wav2vec2_model_fn=wav2vec2_model_fn,
    json_path="modelpaths.json",
    device=device
)

pipeline = PredictionPipeline(
    models=models,
    genre_map=genre_map,
    batch_size=32, 
    ratio=0.25
)

MESSY_DATA_DIR = Path(r'Z:\dlgen-AI-project\data')

test_df = pd.read_csv(MESSY_DATA_DIR / "test.csv") 

result = pipeline.predict_dataset(
    test_df = test_df,
    data_dir = MESSY_DATA_DIR
)
result.to_csv("output.csv")
