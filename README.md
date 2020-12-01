# AIStockSentiment

## Objective/Summary
The Objective of this is to be able pull "Sentiment" of a stock from news articles and snippets. In order to do this, it uses a NLU Network and Tokeniser to determine wether the stock has a good overall sentiment, or a bad overall sentiment. This can be determined when the sentiment is negative, its a genraly bad stock, and if the sentiment is positive, its a generaly good stock. How much negative or positive can tell how bad or good the stock generaly is. 

## API Request
First, we have to get our data. We are doing this by use of an Polygon.io API for news as our first source, and NewsAPI as a second API data source. We are using external modules for both of these APIs. We have provided an example in Python3 below.
```python
from newsapi import NewsApiClient
import alpaca_trade_api as tradeapi

polygonapi = tradeapi.REST('SECRETAPIKEY','https://api.polygon.io' )
newsapi = NewsApiClient(api_key='ANOTHERSECRETAPIKEY')
```

## Determining The Token Of A Sentence
There are two types of objects that are central to this library, namely the `Sentence` and `Token` objects. A
`Sentence` holds a textual sentence and is essentially a list of `Token`. For example, if we were to run the very simple sentiment analysis on the sentence `The grass is green`, it will have a token of 5. This tells us that the sentence consists of 5 tokens. You can access the tokens of a sentence via their token id or with their index in python. This determines the tokenisation of the sentence, and is not very useful as we are looking for sentiment, aka `Sentence` not `Token`, but it can still be used as a validity check for our end result. Below I provided a Python3 example of tokenization on a snippet of AAPL news. 
```python
from flair.data import Sentence

sentence = Sentence('All-time highs are all the rage in 2019. Leading the pack, is the world’s most valuable company, Apple (AAPL). The tech giant’s share price added further muscle by closing December 17’s session at a new record high of $280.41 per share.The latest nudge upward came following news of the most recent détente')

print(sentence)
```
The output was a tokenization of 62. This tells us that the stock AAPL, (The one we got news for), current has a sentiment "value" of 62. Lets say we check the tokenization again tommorow, and it is 120, then we will not to buy this stock because the volatility is way too high. The value of the tokenization is saved to a .csv file every time someone runs an analysis. Then, it can analyze the previous data to automaticaly effect the given sentiment.

## Loading The Corpus
One of the first stops toward the actuall predicting is loading the english corpus. The corpus can be used as a data base of pretrained models, so we do not have to waste time training the models. Its also a good idea to use previous databses/corpuses because they have been selectivley trained so that we have maximum accuracy. But, if you are not satisfied, you can do training using the current corpus, and make an even more accurate one, even though I would not suggest it. Below is an example to load a Flair Data Corpus into Python3.
```python
import flair

model = flair.models.TextClassifier.load('en-sentiment')
```

## Generating Sentiment
The really cool thing about AI is that it takes a suprising small amount of code. The reason for this is becasue all of the "code" is saved in .pb files in which the AI is constantly changing. It really saves the algorithm inide these .pb files, therefore, I do not ahve to code an algorith to evaluate sentiment. We can use the pretrained corpus we loaded in the last step. This corpus is an already trained algorithm that we can use. And in case you are wondering where it saves the algorithm, it always does this line before it starts:
```shell
2020-12-01 09:10:50,546 loading file C:\Users\Mohit\.flair\models\sentiment-en-mix-distillbert_3.1.pt
```
This pretty much means that it is loading its pretrained algorithm that it generated the last time that we used it. This means that every time you use the tool, it will becaome smarter and faster the predicting sentiment. This is the power of AI. 

## Evaluating Sentiment
Now, we have the model loaded and the algorithm trained, now we can start evaluating the model. First, we pull all data from the API endpoints. Then we get all news snippets that have the stock name in it or relates to its market. Next, we can save all of these news to a list, each sentence a new char in the list. Then, we iterate through the list and then use our generated model to evaluate it and every time it sees a positive, it add how much positive it is to a variable, and same vice versa. Below I have gine a Python3 example of this for you to better visualize what I am saying.
```python
response = newsapi.get_everything(qintitle=stock)

news = api.polygon.news(stock)

file = open('news.txt', 'w')

sentiment = 0

for line in response['articles']:
    words = str(line['title'])
    file.write(words)
    sentence = Sentence(str(words))
    model.predict(sentence)
    total_sentiment = sentence.labels
    print(str(words))

    if total_sentiment[0].value == 'NEGATIVE':
        print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
        sentiment -= total_sentiment[0].to_dict()['confidence'] / 2  # Flair favors negative outcomes

    # Checks to see if the sentiment is positive and adds how positive flair thinks it is
    elif total_sentiment[0].value == 'POSITIVE':
        print(str(total_sentiment[0].value) + " : " + str(total_sentiment[0].to_dict()['confidence']))
        sentiment += total_sentiment[0].to_dict()['confidence']
```
We run 2 of the above scripts to determine sentiment. One for NewsAPI, and one for Polygon.io. We also ran another script for the amrket, in this case AAPL, we ran another script on nasdaq market, and added that to the total sentiment

## Results
The results were mildly confusing to me, as I thought that the stock was doing well, but when I ran news analysis on it, it gave me `-6.000672280788422`. This was hard to belive at first, but when I waited a few days and expected it to go up, it actually crashed! The AAPL Stock crashed 15 points in the Nasdaq market. The current date of writing this is December 1, 2020, at 10:49:56. I have also linked a demo below of my code in action in case you would like to see that.

https://youtu.be/YABkU2FsqZA
