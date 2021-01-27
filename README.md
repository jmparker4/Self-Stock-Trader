# Self-Stock-Trader

You can read more about the creatation of this project at:https://jdparkerjake.medium.com/my-experience-creating-a-simple-trading-bot-b4cadee728c9

Exacutive Summary: 
Throughout my recent months, I have come across more and more news articles about how hedge funds and banks are building massive Auto-trading AIs. Therefore, I felt inspired to attempt to create my own self-trading AI. My goal for this this project was to create a self-updating price predicting AI for a single stock. I ended up pulling my data using an open source API called yFinance; which can be found here: https://pypi.org/project/yfinance/. I originally went in building a linear model to project stock prices in 5 minute intervals up to 30 minutes into the future. Halfway through creating this model I also began creating a SVC that would recommend if a stock should be purchased based off your current portfolios holdings (**This is still under development**). My accuracy score for my linear model ranged between 92%-96% depending on my current data set. I believe this was happening for two reasons; one I was only predicting a change in a 5-minute stock’s price and due to having a slightly overfit model. After having the model created, I began working on a self-updating dataset and predictions. This was achieved by having my data being removed whenever a new piece of data was being added.

Risks/Limitations and Assumptions:
There are and were several different assumptions made when creating this model. The model is assuming there is no market risk, fundamental risk or environmental risks when predicting the stock price. There are also several limitations when I as creating the model; I was unable to create strong Ais such as neural networks due to technical limitations with my current computing setup.

Next steps:
I would like to have my model become a fully executable program that can run in the background and not have a it run through a python script though the command line. I am also currently working on better optimizing my model by changing the technical metrics within it and try other modeling types.


Required libraries to run:
Sklearn
Pandas
Numpy
Yfinance
Time
Datetime
TA
