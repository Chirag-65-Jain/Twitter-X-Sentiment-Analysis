# Twitter-X-Sentiment-Analysis
***
_**This is an University project**_

This project aims to provide insight into the emotions expressed via tweets during the COVID-19 period of 2021.

We used the VADER sentiment tool for analysis, which gave us an accuracy of 88% when the emotions were categorized into "Positive", "Neutral" and "Negative". This tool is specifically curated to analyse the sentiments expressed on social media platforms, such as Twitter (now X). 

We mixed multiple datasets that we acquired from [Kaggle](https://kaggle.com/datasets) and ended up with an excel file with 200000+ records for analysis. After data cleaning, the file was analysed using VADER and two new fields were added to file: Sentiment Score and Sentiment. The file was then saved in the same directory in .csv format.

Data visualisation was carried out after the file was saved and a line chart was plotted month-wise and yearly.

All of these functions were accessed via a window which popped upon th execution of the program, that allows the user to enter the path of the file to be analysed and visualise.
***
Packages:
  * tkinter
  * Pandas
  * Natural language toolkit(nltk)
  * spacy
  * sklearn
  * matplotlib
  * seaborn
  * contractions
  * re
  * sos

References: 
  * [VADER Documentation](https://github.com/cjhutto/vaderSentiment)
  
