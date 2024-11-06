import os
import time
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

def generate_text(model, tokenizer, prompt, device, max_new_tokens, min_new_tokens=40):
    """
    Функция для генерации текста с использованием модели и токенизатора.

    :param model: Загрузенная модель для генерации текста.
    :param tokenizer: Токенизатор, соответствующий модели.
    :param prompt: Строка с инструкциями и контекстом для генерации.
    :param device: Устройство (CPU или GPU), на котором выполняется генерация.
    :param max_new_tokens: Максимальное количество генерируемых новых токенов.
    :param min_new_tokens: Минимальное количество генерируемых новых токенов.
    :return: Сгенерированный текст (только резюме).
    """
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        if tokenizer.pad_token_id is not None:
            attention_mask = (inputs != tokenizer.pad_token_id).long()
        else:
            attention_mask = None

        generated_ids = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Определяем метку, после которой начинается резюме
        summary_marker = "Резюме:"

        if summary_marker in generated_text:
            # Извлекаем текст после метки "Резюме:"
            summary = generated_text.split(summary_marker, 1)[1].strip()
            return summary
        else:
            # Если метка не найдена, возвращаем весь сгенерированный текст
            return generated_text.strip()

    except Exception as e:
        logger.error(f"Ошибка при генерации текста: {e}")
        return ""

def split_text(text, max_tokens=1000):
    """
    Разбивает текст на сегменты с максимальным количеством токенов.

    :param text: Входной текст.
    :param max_tokens: Максимальное количество токенов в сегменте.
    :return: Список текстовых сегментов.
    """
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        current_segment.append(word)
        current_length += 1
        if current_length >= max_tokens:
            segments.append(" ".join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments

def load_tokenaizer(model_name):
    local_path = '/root/.cache/whisper/tokenizer'

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        logger.info(f"Токенайзер из кэша {local_path}")
        return tokenizer
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            token=HUGGINGFACE_TOKEN
        )
        tokenizer.save_pretrained(local_path)
        logger.info(f"Токенайзер скачал из интернета и сохранил в {local_path}")
        return tokenizer

def load_llm(model_name):
    local_path = '/root/.cache/whisper/llm'

    try:
        tokenizer = AutoModelForCausalLM.from_pretrained(local_path)
        logger.info(f"LLM модель из кэша {local_path}")
        return tokenizer
    except:
        tokenizer = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float32,  # Используем float32 для CPU
            low_cpu_mem_usage=True
        )
        tokenizer.save_pretrained(local_path)
        logger.info(f"LLM модель скачал из интернета и сохранил в {local_path}")
        return tokenizer

# Загрузка моделей при запуске приложения для оптимизации производительности
@app.on_event("startup")
def load_models():
    global whisper_model, tokenizer, model_llm, device_llm

    # Загружаем модель Whisper
    logger.info("Загрузка модели Whisper...")
    whisper_model = whisper.load_model("large")

    # Название модели на Hugging Face Hub
    model_name = "google/gemma-2-2b-it"


    # Определяем устройство (CPU или GPU). Пока что только CPU установлен
    # device_llm = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device_llm = torch.device("cpu")
    logger.info(f"Используемое устройство для LLM: {device_llm}")


    # Загрузка токенизатора и модели из Hugging Face Hub
    logger.info("Загрузка токенизатора и модели gemma-2-2b-it...")
    tokenizer = load_tokenaizer(model_name)
    model_llm = load_llm(model_name).to(device_llm)

    # Проверяем и устанавливаем pad_token_id, если он не установлен
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.info("Установка pad_token_id в eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Если eos_token_id также не установлен, устанавливаем pad_token_id вручную
            pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
            if pad_token_id is not None:
                tokenizer.pad_token = "[PAD]"
                tokenizer.pad_token_id = pad_token_id
                logger.info("Установка pad_token_id в '[PAD]'")
            else:
                raise ValueError("Не удалось определить pad_token_id. Пожалуйста, установите его вручную.")

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
                    fp16=torch.cuda.is_available()
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Объединяем транскрипции
        full_transcription = "\n".join(transcriptions)
        logger.info("Полная транскрипция завершена.")

        # Разбиваем транскрипцию на сегменты для резюмирования
        logger.info("Разбиение транскрипции на сегменты для резюмирования...")
        text_segments = split_text(full_transcription, max_tokens=1000)

        # Генерируем резюме для каждого сегмента
        summaries = []
        for idx, segment in enumerate(text_segments, 1):
            logger.info(f"Генерация резюме для сегмента {idx}/{len(text_segments)}")
            summary_prompt = f"""
Сократите следующий разговор до 2-3 предложений, сохранив основную информацию и ключевые моменты. Резюме должно быть на русском языке.

Разговор:
{segment}

Резюме:
"""
            summary = generate_text(
                model=model_llm,
                tokenizer=tokenizer,
                prompt=summary_prompt,
                device=device_llm,
                max_new_tokens=150,
                min_new_tokens=40
            )
            summaries.append(summary)

        # Объединяем все резюме в одно итоговое резюме
        final_summary = "\n".join(summaries)
        logger.info("Итоговое резюме создано.")

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
                "summary": final_summary
            }
        })

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return JSONResponse(status_code=500, content={
            "success": 0,
            "error": str(e),
            "data": {}
        })
#uvicorn transcribe_and_summarize_gemma_V5_with_endpoint_V1:app --host 0.0.0.0 --port 8000 --workers 1
