from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from database import db_manager
from main import predict_batch, predict_toxicity

app = FastAPI(title="Toxicity Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommentPrediction(BaseModel):
    comment_id: int
    comment_text: str
    comment_toxic: int
    predicted_probability: float
    predicted_class: int
    is_correct: bool


class SingleCommentRequest(BaseModel):
    text: str


class SingleCommentResponse(BaseModel):
    text: str
    probability: float
    predicted_class: int


class SaveCommentRequest(BaseModel):
    text: str
    toxicity: int


class SaveCommentResponse(BaseModel):
    comment_id: int
    message: str


class RecentComment(BaseModel):
    comment_id: int
    comment_text: str
    comment_toxic: int


@app.get("/")
async def root():
    return {"message": "Toxicity Classifier API работает"}


@app.get("/predict-all", response_model=List[CommentPrediction])
async def predict_all():
    """Предсказать токсичность для всех комментариев"""
    comments = db_manager.get_all_comments()

    if not comments:
        raise HTTPException(status_code=404, detail="Комментарии не найдены")

    texts = [c['comment_text'] for c in comments]
    predictions = predict_batch(texts)

    results = []
    for i, comment in enumerate(comments):
        prob = predictions[i] if i < len(predictions) else 0.5
        pred_class = 1 if prob > 0.5 else 0

        results.append({
            "comment_id": comment['comment_id'],
            "comment_text": comment['comment_text'],
            "comment_toxic": comment['comment_toxic'],
            "predicted_probability": float(prob),
            "predicted_class": pred_class,
            "is_correct": (comment['comment_toxic'] == pred_class)
        })

    return results


@app.post("/predict-single", response_model=SingleCommentResponse)
async def predict_single(request: SingleCommentRequest):
    """Предсказать токсичность для одного комментария"""
    try:
        text = request.text

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Текст комментария не может быть пустым")

        probability = predict_toxicity(text)
        predicted_class = 1 if probability > 0.5 else 0

        return {
            "text": text,
            "probability": float(probability),
            "predicted_class": predicted_class
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.post("/save-comment", response_model=SaveCommentResponse)
async def save_comment(request: SaveCommentRequest):
    """Сохранить комментарий в бд"""
    try:
        text = request.text
        toxicity = request.toxicity

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Текст комментария не может быть пустым")

        if toxicity not in [0, 1]:
            raise HTTPException(status_code=400, detail="Токсичность должна быть 0 или 1")

        comment_id = db_manager.save_comment(text, toxicity)

        return {
            "comment_id": comment_id,
            "message": "Комментарий успешно сохранен"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении: {str(e)}")



@app.get("/health")
async def health_check():
    db_status = "connected" if db_manager.connection and db_manager.connection.is_connected() else "disconnected"

    from main import model_instance
    model_status = "loaded" if model_instance.model is not None else "not_loaded"

    return {
        "status": "ok",
        "database": db_status,
        "model": model_status
    }


def start():
    """Запуск сервера"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()