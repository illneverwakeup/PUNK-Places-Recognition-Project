import os
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile
import cv2
from punk_classifier import make_prediction

with open('tg_token.txt', encoding='utf-8') as f:
    TOKEN: Final = str(f.readline()[:-1])
BOT_USERNAME: Final = '@PUNK_Places_bot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Отправь мне изображение.')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Help!')


async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Custom!')


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await update.message.photo[-1].get_file()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        print(temp_file_path)
        await photo_file.download_to_drive(temp_file_path)

    try:
        result, score = make_prediction(temp_file_path)
        await update.message.reply_text(f"Это {result}! Уверен на {score*100 :.2f}%!")
    finally:
        os.remove(temp_file_path)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text


if __name__ == '__main__':
    print('Bot started...')
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling(poll_interval=1, )

    print('Bot polling...')
    application.run_polling(poll_interval=2)
