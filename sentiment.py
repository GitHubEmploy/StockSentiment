#IMPORTING DEPENDENCIES
print('Importing Flair/Torch...')
import flair
import torch
from flair.data import Sentence

print('Importing WSS(s)...')
from newsapi import NewsApiClient
import alpaca_trade_api as tradeapi
#DONE IMPORTING DEPENDENCIES

#SET RUN DEVICE AS CPU
flair.device = torch.device('cpu')

#STATING POLYGON.IO API
api = tradeapi.REST('ZFA3HZJSCNVu7dJGm0Y4pNIBjjIRRg4c','https://api.polygon.io' )

#DEFINING FUNCTION
def sentiment(stock, api):
    #LOADING TRADERVIEW
    url = 'https://www.tradingview.com/screener/'

    #LOADING FLAIR
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

    #NEWSAPI API call
    newsapi = NewsApiClient(api_key='c9f2ed7ab2724a699b000dd683262960')

    #GET THE ARTICLES
    response = newsapi.get_everything(qintitle=stock)

    #SPECIFY API CALL INSIDE THE FUNCTION
    news = api.polygon.news(stock)

    #OPEN NEWS.TXT TO WRITE NEWS
    file = open('news.txt', 'w')

    #VERIFY SENTIMENT VARIABLE IS 0
    sentiment = 0
    print(response)
    #ITERATES THROUGH EVERY NEWS ARTICLE FROM NEWS API
    for line in response['articles']:
        words = str(line['title'])
        file.write(words)
        #RUNS FLAIR SENTIMENT ANALYSIS
        sentence = Sentence(str(words))
        flair_sentiment.predict(sentence)
        total_sentiment = sentence.labels
        print(str(words))

        # Checks to see if the sentiment is negative and subtracts by how negative flair thinks it is
        if total_sentiment[0].value == 'NEGATIVE':
            print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
            sentiment -= total_sentiment[0].to_dict()['confidence'] / 2  # Flair favors negative outcomes

        # Checks to see if the sentiment is positive and adds how positive flair thinks it is
        elif total_sentiment[0].value == 'POSITIVE':
            print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
            sentiment += total_sentiment[0].to_dict()['confidence']

    #ITERATES THROUGH EVERY NEWS ARTICLE FROM POLYGON.IO
    for source in news:
        words = source.summary
        try:
            file.write(words)
        except:
            print('FAILSAFE ACTIVATED')
        file.write('\n')

        # Runs Flair sentiment analysis
        sentence = Sentence(str(words))
        try:
            flair_sentiment.predict(sentence)
        except:
            print("\n")
        total_sentiment = sentence.labels
        print(str(words))

        # Checks to see if the sentiment is negative and subtracts by how negative flair thinks it is
        if total_sentiment[0].value == 'NEGATIVE':
            print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
            sentiment -= total_sentiment[0].to_dict()['confidence'] / 2  # Flair favors negative outcomes

        # Checks to see if the sentiment is positive and adds how positive flair thinks it is
        if total_sentiment[0].value == 'POSITIVE':
            print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
            sentiment += total_sentiment[0].to_dict()['confidence']


    file.close()
    print('Total sentiment', sentiment) #News Sentiment

sentiment('AAPL', api)
