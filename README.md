# StockSentiment

## Objective/Summary

The Objective of this is to be able pull "Sentiment" of a stock from news articles and snippets. In order to do this, it uses a NLU Network and Tokeniser to determine wether the stock has a good overall sentiment, or a bad overall sentiment. This can be determined when the sentiment is negative, its a genraly bad stock, and if the sentiment is positive, its a generaly good stock. How much negative or positive can tell how bad or good the stock generaly is. 

## Determining The Token Of A Sentence
There are two types of objects that are central to this library, namely the `Sentence` and `Token` objects. A
`Sentence` holds a textual sentence and is essentially a list of `Token`. For example, if we were to run the very simple sentiment analysis on the sentence `The grass is green`, it will have a token of 5. This tells us that the sentence consists of 5 tokens. You can access the tokens of a sentence via their token id or with their index in python. This determines the tokenisation of the sentence, and is not very useful as we are looking for sentiment, aka `Sentence` not `Token`, but it can still be used as a validity check for our end result. Below I provided a Python3 example of tokenization on a snippet of AAPL news. 
```python
from flair.data import Sentence

sentence = Sentence('All-time highs are all the rage in 2019. Leading the pack, is the world’s most valuable company, Apple (AAPL). The tech giant’s share price added further muscle by closing December 17’s session at a new record high of $280.41 per share.The latest nudge upward came following news of the most recent détente')

print(sentence)
```
The output was a tokenization of 62. This tells us that the stock AAPL, (The one we got news for), current has a sentiment "value" of 62. Lets say we check the tokenization again tommorow, and it is 120, then we will not to buy this stock because the volatility is way too high. The value of the tokenization is saved to a .csv file every time someone runs an analysis. 

## Loading The Corpus
One of the first stops toward the actuall predicting is loading the english corpus. The corpus can be used as a data base of pretrained models, so we do not have to waste time training the models. Its also a good idea to use previous databses/corpuses because they have been selectivley trained so that we have maximum accuracy. But, if you are not satisfied, you can do training using the current corpus, and make an even more accurate one, even though I would not suggest it.

