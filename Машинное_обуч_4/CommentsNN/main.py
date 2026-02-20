import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

POLITICS_EXCEPTIONS = [
    'интересуюсь политикой',
    'слежу за политикой',
    'изучаю политику',
    'политическая наука',
    'политический аналитик',
    'политика государства',
    'внутренняя политика',
    'внешняя политика',
    'экономическая политика',
    'социальная политика',
    'политическая ситуация',
    'политические события',
    'политическая система',
    'политический строй',
    'политическая жизнь',
    'политическая деятельность',
    'политическая партия',
    'политический курс',
    'политическая власть',
    'политическая элита'
]

morph = pymorphy3.MorphAnalyzer()


def get_russian_stopwords():
    """Получить список русских стоп-слов"""
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['т.д.', 'т', 'д', 'это', 'который', 'которые', 'которых',
                              'свой', 'своём', 'всем', 'всё', 'её', 'оба', 'ещё',
                              'должный', 'должные', 'должных'])
    return russian_stopwords


def remove_othersymbol(text, st_char='\xa0—'):
    """Удаление специальных символов"""
    return ''.join([ch if ch not in st_char else ' ' for ch in text])


def remove_punctuation(text, custom_punctuation=None):
    """Удаление пунктуации"""
    if custom_punctuation is None:
        custom_punctuation = string.punctuation + '«»'
    return ''.join([ch for ch in text if ch not in custom_punctuation])


def remove_numbers(text):
    """Удаление чисел"""
    return ''.join([i if not i.isdigit() else ' ' for i in text])


def remove_multiple_spaces(text):
    """Удаление множественных пробелов"""
    return re.sub(r'\s+', ' ', text, flags=re.I)


def remove_stopwords(text, stopwords_list=None):
    """Удаление стоп-слов"""
    if not isinstance(text, str):
        return text

    if stopwords_list is None:
        stopwords_list = get_russian_stopwords()

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]

    return ' '.join(filtered_words)


def lemmatize_text(text):
    """Лемматизация текста"""
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        tokens = word_tokenize(text)
        lemm_words = []
        for word in tokens:
            try:
                p = morph.parse(word)[0]
                lemm_words.append(p.normal_form)
            except Exception as e:
                print(f"Ошибка при лемматизации слова '{word}': {e}")
                lemm_words.append(word)
        return " ".join(lemm_words)
    except Exception as e:
        print(f"Ошибка при лемматизации: {e}")
        return text


def tokenize_text(text):
    """Токенизация текста"""
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        tokens = word_tokenize(text)
        stopwords_list = get_russian_stopwords()
        tokens = [token for token in tokens if token.lower() not in stopwords_list]
        return " ".join(tokens)
    except Exception as e:
        print(f"Ошибка при токенизации: {e}")
        return text


def preprocess_text_pipeline(text, show_steps=False):
    if not isinstance(text, str) or not text.strip():
        return "" if not show_steps else {"error": "Пустой текст"}

    steps = {}
    current_text = text

    if show_steps:
        steps['original'] = current_text

    current_text = remove_othersymbol(current_text)
    if show_steps:
        steps['after_remove_othersymbol'] = current_text

    current_text = remove_punctuation(current_text)
    if show_steps:
        steps['after_remove_punctuation'] = current_text

    current_text = remove_numbers(current_text)
    if show_steps:
        steps['after_remove_numbers'] = current_text

    current_text = remove_multiple_spaces(current_text)
    if show_steps:
        steps['after_remove_spaces'] = current_text

    stopwords_list = get_russian_stopwords()
    current_text = remove_stopwords(current_text, stopwords_list)
    if show_steps:
        steps['after_remove_stopwords'] = current_text

    current_text = lemmatize_text(current_text)
    if show_steps:
        steps['after_lemmatization'] = current_text

    current_text = tokenize_text(current_text)
    if show_steps:
        steps['final'] = current_text

    if show_steps:
        return steps
    return current_text.strip()


class ToxicityModel:
    def __init__(self, model_path='model.keras', vectorizer_path='vectorizer.pkl'):
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.max_len = 5000
        self.load_model(model_path)
        self.load_vectorizer(vectorizer_path)

    def load_model(self, model_path):
        """Загрузка модели"""
        try:
            if not os.path.exists(model_path):
                print(f"Файл модели {model_path} не найден!")
                self.model = None
                return

            self.model = tf.keras.models.load_model(model_path)

            if hasattr(self.model, 'input_shape'):
                expected_len = self.model.input_shape[1]
                if expected_len is not None:
                    self.max_len = expected_len

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            self.model = None

    def load_vectorizer(self, vectorizer_path):
        """Загрузка векторизатора"""
        try:
            if not os.path.exists(vectorizer_path):
                print(f"Файл векторизатора {vectorizer_path} не найден!")
                print("Будет использована простая токенизация")
                self.vectorizer = None
                return

            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

        except Exception as e:
            print(f"Ошибка при загрузке векторизатора: {e}")
            self.vectorizer = None

    def preprocess_text(self, texts):
        """Предобработка текста с использованием загруженного векторизатора"""
        if isinstance(texts, str):
            texts = [texts]

        processed_texts = []
        for text in texts:
            processed = preprocess_text_pipeline(text)
            processed_texts.append(processed)

        if self.vectorizer is not None:
            try:
                if hasattr(self.vectorizer, 'transform'):
                    vectors = self.vectorizer.transform(processed_texts)
                    sequences = vectors.toarray()

                    if sequences.shape[1] < self.max_len:
                        padding = np.zeros((sequences.shape[0], self.max_len - sequences.shape[1]))
                        sequences = np.hstack([sequences, padding])
                    elif sequences.shape[1] > self.max_len:
                        sequences = sequences[:, :self.max_len]

                    return sequences.astype(np.float32)

                elif hasattr(self.vectorizer, 'texts_to_sequences'):
                    sequences = self.vectorizer.texts_to_sequences(processed_texts)
                    padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
                    return padded

                else:
                    print(f"Неизвестный тип векторизатора: {type(self.vectorizer)}")
                    raise ValueError("Неподдерживаемый тип векторизатора")

            except Exception as e:
                print(f"Ошибка при использовании векторизатора: {e}")
                print("Используем простую токенизацию")
                self.vectorizer = None

        sequences = []
        for text in processed_texts:
            words = text.lower().split()
            seq = [hash(word) % 10000 for word in words]
            sequences.append(seq)

        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded

    def predict(self, texts):
        """Предсказание токсичности"""
        if self.model is None:
            print("Модель не загружена, возвращаем случайные значения")
            if isinstance(texts, str):
                return 0.5
            else:
                return [0.5] * len(texts)

        try:
            processed = self.preprocess_text(texts)
            predictions = self.model.predict(processed, verbose=0)

            if len(predictions.shape) > 1:
                if predictions.shape[1] > 1:
                    predictions = predictions[:, 1]
                else:
                    predictions = predictions.flatten()
            else:
                predictions = predictions.flatten()

            predictions = predictions.tolist()
            return predictions[0] if isinstance(texts, str) else predictions

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            return 0.5 if isinstance(texts, str) else [0.5] * len(texts)


model_instance = ToxicityModel()


def get_model():
    """Получить экземпляр модели"""
    return model_instance


def is_toxic_with_context(text, probability):
    """Определяет, должен ли комментарий считаться токсичным с учетом контекста"""
    text_lower = text.lower()

    politics_words = ['политика', 'политике', 'политику', 'политикой', 'политический', 'политическая']
    has_politics = any(word in text_lower for word in politics_words)

    if not has_politics:
        return probability > 0.5

    for exception in POLITICS_EXCEPTIONS:
        if exception in text_lower:
            print(f"Найдено исключение: '{exception}'")
            return probability > 0.5

    if probability < 0.5:
        print(f"Слово 'политика' в потенциально токсичном контексте, повышаем вероятность")
        return True

    return probability > 0.5


def predict_toxicity(text):
    """Функция для предсказания токсичности одного текста с учетом контекста"""
    probability = model_instance.predict(text)

    is_toxic = is_toxic_with_context(text, probability)

    if is_toxic and probability < 0.5:
        return 0.6
    elif not is_toxic and probability > 0.5:
        return 0.4
    else:
        return probability


def predict_batch(texts):
    """Функция для пакетного предсказания с учетом контекста"""
    probabilities = model_instance.predict(texts)

    for i, text in enumerate(texts):
        is_toxic = is_toxic_with_context(text, probabilities[i])

        if is_toxic and probabilities[i] < 0.5:
            probabilities[i] = 0.6
        elif not is_toxic and probabilities[i] > 0.5:
            probabilities[i] = 0.4

    return probabilities


def predict_toxicity_original(text):
    """Оригинальная функция для предсказания токсичности одного текста без контекста"""
    return model_instance.predict(text)



def predict_batch_original(texts):
    """Оригинальная функция для пакетного предсказания без контекста"""
    return model_instance.predict(texts)


def get_preprocessing_steps(text):
    """Получить все шаги предобработки для отображения"""
    return preprocess_text_pipeline(text, show_steps=True)