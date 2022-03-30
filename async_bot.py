import asyncio
import aiogram
import logging
import sq_track_class

TOKEN = '5184695285:AAGXsJucG2G9gktI3MFVhNZ77vzs7xXzo-k'


bot = aiogram.Bot(token=TOKEN)
dp = aiogram.Dispatcher(bot)

logging.basicConfig(level=logging.INFO)
name = None
pushups = 0
squats = 0
@dp.message_handler(commands='block')
async def cmd_block(message: aiogram.types.Message):
    await asyncio.sleep(10.0)
    await message.reply('Бан')

@dp.message_handler(commands='info')
async def starter(message: aiogram.types.Message):
    await message.reply('''введите команду /train для начала тренировки, затем введите свое имя
    ''') 

@dp.message_handler(content_types=[aiogram.types.ContentType.VIDEO])
async def video(message: aiogram.types.Message):
    global pushups, squats
    reworker = sq_track_class
    video = await message.video.download()
    pu, sq =  reworker.all_track(video.name)
    pushups += pu
    squats += sq
    await message.reply('кол-во прис:' + str(sq) + '\n' + 'кол-во отж:' + str(pu))
    await message.reply(f'общее количество отжиманий у {name}: {pushups} ')
    await message.reply(f'общее количество приседаний у {name}: {squats} ')

@dp.message_handler(commands='train')
async def train(message: aiogram.types.Message):
    await message.reply('Введите имя')

@dp.message_handler(content_types=[aiogram.types.ContentType.TEXT])
async def name(message: aiogram.types.Message):
    global name
    name = message.text
    if name in ['вова', 'витя', 'игорь']:
        await message.reply(name + ', отправьте видео с выполнением 3 отжиманий' + 
                            '\nи трех приседаний')
    else:
        await message.reply('вас нет в базе:(')

if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)


