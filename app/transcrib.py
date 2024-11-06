import os
import time
import whisper
from dotenv import load_dotenv  # Для загрузки переменных окружения
import logging
import librosa
import soundfile as sf
import tempfile
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные из .env файла
load_dotenv()

# Получаем токен из переменной окружения
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Проверяем, установлен ли токен (необходим для закрытых моделей)
if not HUGGINGFACE_TOKEN:
    raise ValueError("Не удалось найти Hugging Face API токен. Убедитесь, что переменная окружения HUGGINGFACE_TOKEN установлена.")

app = FastAPI(title="Audio Recognition API")

def amplify_audio(input_audio_path, gain_db=5.0, sr=16000):
    """
    Усиливает аудиофайл на заданное количество децибел (dB).

    :param input_audio_path: Путь к исходному аудиофайлу.
    :param gain_db: Уровень усиления в децибелах.
    :param sr: Частота дискретизации для загрузки аудио.
    :return: Путь к усиленному аудиофайлу.
    """
    try:
        logger.info(f"Загрузка аудиофайла для усиления: {input_audio_path}")
        y, sr = librosa.load(input_audio_path, sr=sr)

        logger.info(f"Исходная длительность аудио: {librosa.get_duration(y=y, sr=sr):.2f} секунд")

        # Перевод децибел в коэффициент усиления
        gain = 10 ** (gain_db / 20)
        logger.info(f"Усиление аудио на {gain_db} dB (коэффициент усиления: {gain:.2f})")

        # Усиление аудио
        y_amplified = y * gain

        # Предотвращение клиппинга (ограничение амплитуды)
        max_amplitude = max(abs(y_amplified))
        if max_amplitude > 1.0:
            logger.info("Нормализация усиленного аудио для предотвращения клиппинга.")
            y_amplified = y_amplified / max_amplitude

        # Создаём временный файл для сохранения усиленного аудио
        fd, output_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Закрываем дескриптор файла

        logger.info(f"Сохранение усиленного аудиофайла: {output_audio_path}")
        sf.write(output_audio_path, y_amplified, sr)

        return output_audio_path

    except Exception as e:
        logger.error(f"Ошибка при усилении аудио: {e}")
        raise

def split_audio(input_audio_path, chunk_length_sec=600, sr=16000):
    """
    Разбивает аудиофайл на чанки заданной длины.

    :param input_audio_path: Путь к исходному аудиофайлу.
    :param chunk_length_sec: Длина одного чанка в секундах (по умолчанию 10 минут).
    :param sr: Частота дискретизации.
    :return: Список путей к чанкам.
    """
    try:
        logger.info(f"Загрузка аудиофайла для разбиения: {input_audio_path}")
        y, sr = librosa.load(input_audio_path, sr=sr)
        total_duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Общая длительность аудио: {total_duration:.2f} секунд")

        chunks = []
        for start in range(0, int(total_duration), chunk_length_sec):
            end = min(start + chunk_length_sec, int(total_duration))
            y_chunk = y[start * sr:end * sr]
            fd, chunk_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # Закрываем дескриптор файла
            sf.write(chunk_path, y_chunk, sr)
            chunks.append(chunk_path)
            logger.info(f"Создан чанк: {chunk_path} ({start} - {end} секунд)")

        return chunks

    except Exception as e:
        logger.error(f"Ошибка при разбиении аудио: {e}")
        raise

# Загрузка моделей при запуске приложения для оптимизации производительности
@app.on_event("startup")
def load_models():
    global whisper_model

    # Загружаем модель Whisper
    logger.info("Загрузка модели Whisper...")
    whisper_model = whisper.load_model("large")


    logger.info("Модели успешно загружены и готовы к использованию.")

@app.post("/audio/recognize")
async def recognize_audio(file: UploadFile = File(...)):
    """
    Endpoint для транскрипции и суммаризации загруженного аудиофайла.

    :param file: Аудиофайл в формате .mp3, .wav, .m4a, .flac.
    :return: JSON с транскрипцией и суммаризацией.
    """
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an audio file (mp3, wav, m4a, flac).")

    try:
        start_time = time.time()

        # Сохраняем загруженный файл во временное хранилище
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
        os.close(fd)  # Закрываем дескриптор файла
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"Сохранён загруженный файл: {temp_path}")

        # Усиливаем аудиофайл
        logger.info("Усиление аудиофайла...")
        amplified_audio_file = amplify_audio(temp_path, gain_db=5.0)

        # Разбиваем усиленный аудиофайл на чанки (по 10 минут)
        logger.info("Разбиение аудиофайла на чанки...")
        chunks = split_audio(amplified_audio_file, chunk_length_sec=600)  # 600 секунд = 10 минут

        # Транскрибируем каждый чанк и объединяем транскрипции
        transcriptions = []
        for chunk in chunks:
            logger.info(f"Транскрибирование чанка: {chunk}")
            try:
                result = whisper_model.transcribe(
                    chunk,
                    language="ru",
                    task="transcribe",
                    verbose=False,
                    fp16=False #torch.cuda.is_available()
                )
                transcriptions.append(result["text"])
                logger.info(f"Транскрипция чанка: {result['text'][:100]}...")  # Логируем первые 100 символов
            except Exception as e:
                logger.error(f"Ошибка при транскрибировании чанка {chunk}: {e}")
                transcriptions.append("")  # Добавляем пустую строку в случае ошибки
            finally:
                # Удаляем временный чанк
                if os.path.exists(chunk):
                    os.remove(chunk)
                    logger.info(f"Удалён временный чанк: {chunk}")
                # Освобождаем память
                if 'result' in locals():
                    del result
                gc.collect()
                # if torch.cuda.is_available():
                    # torch.cuda.empty_cache()

        # Объединяем транскрипции
        full_transcription = "\n".join(transcriptions)
        logger.info("Полная транскрипция завершена.")


        # Рассчитываем время обработки
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Время обработки запроса: {elapsed_time:.2f} секунд")

        # Удаляем временные файлы
        if os.path.exists(amplified_audio_file):
            os.remove(amplified_audio_file)
            logger.info(f"Удалён временный усилённый аудиофайл: {amplified_audio_file}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Удалён временный загруженный файл: {temp_path}")

        return JSONResponse(status_code=200, content={
            "success": 1,
            "error": "",
            "data": {
                "text": full_transcription,
                "summary": "todo",
                "elapsed": elapsed_time
            }
        })

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return JSONResponse(status_code=500, content={
            "success": 0,
            "error": str(e),
            "data": {}
        })
# uvicorn transcribe_and_summarize_gemma_V5_with_endpoint_V1:app --host 0.0.0.0 --port 8000 --workers 1
# python -m uvicorn transcrib:app --host 0.0.0.0 --port 8000 --workers 1