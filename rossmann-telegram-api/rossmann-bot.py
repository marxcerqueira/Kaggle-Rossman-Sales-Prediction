import os
import pandas as pd
import json
import requests
import telegram
import matplotlib.pyplot as plt
import seaborn           as sns
from flask import Flask, request, Response 
from io import BytesIO

# constants
TOKEN  = '1772058988:AAEy1JExctBfpdXfA9E6AtfP_VG_zmSuvyQ'
TOKEN2 = '1970888326:AAGojEWanr2-RQb3hYHaGwIxfmidi3qoGoU'

# # info about the Bot
# https://api.telegram.org/bot1970888326:AAGojEWanr2-RQb3hYHaGwIxfmidi3qoGoU/getMe

# # get updates
# https://api.telegram.org/bot1970888326:AAGojEWanr2-RQb3hYHaGwIxfmidi3qoGoU/getUpdates

# # Webhook
# https://api.telegram.org/bot1970888326:AAGojEWanr2-RQb3hYHaGwIxfmidi3qoGoU/setWebhook?url=https://cc0a2c7a371f9b.localhost.run/

# # Webhook Heroku
# https://api.telegram.org/bot1772058988:AAEy1JExctBfpdXfA9E6AtfP_VG_zmSuvyQ/setWebhook?url=https://rossman-bot-telegram.herokuapp.com/


# # send message
# https://api.telegram.org/bot1970888326:AAGojEWanr2-RQb3hYHaGwIxfmidi3qoGoU/sendMessage?chat_id=669387603&text=Hi Marx, I am doing good, tks!
# 669387603

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format( TOKEN2 )
    url = url + 'sendMessage?chat_id={}'.format(chat_id)
    
    r = requests.post(url, json = {'text': text})
    print('Status Code{}'.format(r.status_code))

    return None

def load_dataset(store_id):
    # loading test dataset
    df10 = pd.read_csv('test.csv', low_memory = False)
    df_store_raw = pd.read_csv('store.csv', low_memory = False)

    # merge test dataset + store attributes dataset
    df_test = pd.merge(df10, df_store_raw, how = 'left', on = 'Store')

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis = 1)

        # convert dataframe to json
        data = json.dumps(df_test.to_dict(orient = 'records'))
    else:
        data = 'error'
        
    return data

def predict(data):
    # API Call
    url = 'https://rossmann-model-pred.herokuapp.com/rossmann/predict'
    header = {'Content-type': 'application/json' } #indica para a api qual tipo de dados que estamos recebendo
    data = data

    r = requests.post(url, data = data, headers = header)
    print('Status Code {}'.format(r.status_code))

    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '') # remove the / that telegram uses

    try:
        store_id = int(store_id)
    
    except ValueError:
        store_id = 'error'

    return chat_id, store_id

# API Initialize
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST']) # create the endpoint/route (from where the user message will come)

def index(): # runs every time the endpoint / is called

    if request.method == 'POST':
        message = request.get_json() # get the json data
        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            #loading data
            data = load_dataset(store_id)

            if data != 'error':
                #prediction
                d1 = predict(data)

                #calculation
                d2 = d1[['store', 'prediction']].groupby('store').sum(0).reset_index()
                
                #send message
                intro = 'Generating sales forecast for store {}...'.format(d2['store'].values[0])
                send_message(chat_id, intro)

                msg = 'Store Number {} will sell R$ {:,.2f} in the next 6 weeks'.format(
                            d2['store'].values[0], 
                            d2['prediction'].values[0])

                # sending graphics with de prediction
                # fig = plt.figure()
                # sns.lineplot(x = 'week_of_year', y = 'prediction', data = d1)
                # plt.title('Weekly Sales FOrecast: Store {}'.format(d2['store'].values[0]))
                # plt.xlabel('Week Year (starting from week 31 - July 19th)')
                # plt.ylabel('Sales Prediction')

                # buffer = BytesIO()
                # fig.savefig(buffer, formart = 'png')
                # buffer.seek(0)
                # Bot.send_photo(chat_id=chat_id, photo = buffer)

                send_message(chat_id, msg)
                return Response('Ok', status = 200)

            else:
                send_message(chat_id, 'Store Not Available for prediction')
                return Response('Ok', status = 200) 

        else:
            send_message(chat_id, 'Wrong Store ID. Please use only integer store numbers')
            return Response('Ok', status = 200)
    else:
        return '<h1> Rossmann Telegram BOT </h1>'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port, debug = True)
