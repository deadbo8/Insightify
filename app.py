import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import requests
import pandas as pd
import os
import time

from flask import Flask
from flask import render_template, request, redirect, url_for
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

plt.style.use('ggplot')

import nltk

# Download if required 
'''nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('stopwords')'''


app = Flask(__name__)  

def get_soup(url):
    r = requests.get('http://localhost:8050/render.html', params=
    {'url': url, 'wait': 3})
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

def  process_url(User_url,page_number):
    split_data = User_url.split("/")
    product_id = split_data[5]
    return 'https://www.amazon.in/product-reviews/'+product_id+'?reviewerType=all_reviews&pageNumber='+str(page_number)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results',methods=['GET'])
def result():
    # Get url from home.html
    url = request.args.get('url')

    path_static=r'C:\Users\aksha\Desktop\amazon_projext\app\static\images'
    path_temp=r'C:\Users\aksha\Desktop\amazon_projext\app\template'

    #make a list called review list
    reviewlist= []
    reviewlist.clear()

    product_name=""

    total_reviews=0
    total_positive=0
    average_rating=0.0

    index=1
    CleanCache(directory=path_static)
 

    for page_number in range(1,3):
        final_url=process_url(url,page_number)
        
        # getting soup of submitted url 
        soup= get_soup(final_url)

        reviews = soup.find_all('div', {'data-hook': 'review'})
    
        try:
            for item in reviews:
                review = {
                'id': index,
                'product': soup.title.text.replace('Amazon.in:Customer reviews:', '').strip(),
                'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
                'rating': float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
                'Raw_reviews': item.find('span', {'data-hook': 'review-body'}).text.strip()
                }
                reviewlist.append(review)
                index +=1
        except:
            pass

        if not soup.find('li', {'class': 'a-disabled a-last'}):
            pass
        else:
            break
            

        print(index)

    # Creating a pandas Dataframe
    df = pd.DataFrame(reviewlist)
    #df.to_excel('/static/Product1.xlsx', index=False)

    product_name= df['product'][1]

    average_rating = df['rating'].mean()

    # Initializing Sentiment Intensiy Analyzer
    sia = SentimentIntensityAnalyzer()

    # Processing Stop Words
    stop_words = set(stopwords.words('english'))

    time.sleep(3)
    df['body'] = df['Raw_reviews'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    
    # Running SentimentIntensityAnalyzer on entire data
    result={}
    for i, row in tqdm(df.iterrows(), total = len(df)):
        body = row['body']
        myid = row['id']
        result[myid] = sia.polarity_scores(str(body))
    
    final_scores = pd.DataFrame(result).T
    final_scores = final_scores.reset_index().rename(columns={'index':'id'})
    final_scores = final_scores.merge(df , how='left')


    # Genrating Graph Sentiments in relation to Amazon Star Rating
    import seaborn as sns 
    ax1 = sns.barplot(data=final_scores, x='rating', y='compound',palette='Blues' )
    ax1.set(xlabel='Star Rating', ylabel='Sentiment Intensity(Higher being more positive)')
    ax1.set_title('Sentiments in relation to Amazon Star Rating')

   

    for i in ax1.containers:
        ax1.bar_label(i,)

    def bar_label(ax1):
        for p in ax1.containers:
            ax1.bar_label(p,)

    graph1 = 'Sentiments in relation to Amazon Star Rating.png'
    file_path = os.path.join(path_static, graph1)
    plt.savefig(file_path)
    plt.clf()


    fig, axs = plt.subplots(1,3, figsize=(15, 5))
    sns.barplot(data= final_scores, x='rating', y='pos',ax= axs[0],palette='Blues' )
    sns.barplot(data= final_scores, x='rating', y='neu',ax=axs[1],palette='Blues' )
    sns.barplot(data= final_scores, x='rating', y='neg',ax=axs[2],palette='Blues' )

    axs[0].set_title('positive')
    axs[1].set_title('neutral')
    axs[2].set_title('negative')
    plt.tight_layout()
    graph2 = 'Sentiments distribution in realtion to star ratings.png'
    file_path = os.path.join(path_static, graph2)
    plt.savefig(file_path)
    plt.clf()


    # changing labels of the upcoming graphs
    final_scores["sentiment_label"] = final_scores["body"].apply(lambda x: "Positive" if sia.polarity_scores(x)["compound"] > 0 else ("Neutral" if sia.polarity_scores(x)["compound"]==0 else "Negative"))

    #count the number of reviews for various parameters
    total_positive = final_scores['sentiment_label'].value_counts()['Positive']
    total_reviews = final_scores['sentiment_label'].value_counts()['Positive'] +final_scores['sentiment_label'].value_counts()['Neutral']+final_scores['sentiment_label'].value_counts()['Negative']
   
   
   
    plt.subplots_adjust(left=0.15, bottom=0.25)
    ax3 = final_scores['sentiment_label'].value_counts().sort_index().plot(kind='barh', color=sns.color_palette("Blues"),figsize=(10,5))
    ax3.set_xlabel('Count of Reviews', fontname="Arial", size=14)
    ax3.set_ylabel('Sentiment Review', fontname="Arial", size=14)
    ax3.set_title('Count of Reviews by Sentiment', fontname="Arial", size=16)
    ax3.grid(visible=True, linestyle='-.')
    ax3.set_yticklabels(ax3.get_yticklabels(), fontname="Arial", fontsize=12, rotation=0)
    ax3.set_facecolor("#f2f2f2")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['left'].set_visible(True)
    ax3.set_axisbelow(True)

    for i, v in enumerate(final_scores['sentiment_label'].value_counts().sort_index()):
        ax3.text(v, i, str(v), color='black', fontweight='bold', fontname="Arial", fontsize=12)
        #ax3.bar(i, v, edgecolor='black',bottom =-0.7, linewidth=1.5, alpha=0.8)
        

    positive_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[1])
    neutral_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[2])
    negative_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[3])
    leg = plt.legend([negative_bar, neutral_bar, positive_bar], ['Positive', 'Neutral', 'Negative'],loc='lower right')
    for text in leg.get_texts():
        text.set_alpha(0.5)
    leg.legendPatch.set_alpha(0.5)
    ax3.axhline(final_scores['sentiment_label'].value_counts().mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.tight_layout()



    graph3 = 'sentiments by review count.png'
    file_path = os.path.join(path_static, graph3)
    plt.savefig(file_path)
    plt.clf()

    
    # Review Rating Count
    plt.subplots_adjust(left=0.15, bottom=0.25)
    ax4 = final_scores['rating'].value_counts().sort_index().plot(kind='barh', color=sns.color_palette("Blues"),figsize=(10,5))
    ax4.set_xlabel('Count of Reviews', fontname="Arial", size=14)
    ax4.set_ylabel('Star Ratings', fontname="Arial", size=14)
    ax4.set_title('Reviews By Star Rating', fontname="Arial", size=16)
    ax4.grid(visible=True, linestyle='-.')
    ax4.set_yticklabels(ax4.get_yticklabels(), fontname="Arial", fontsize=12, rotation=0)
    ax4.set_facecolor("#f2f2f2")
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(True)
    ax4.spines['left'].set_visible(True)
    ax4.set_axisbelow(True)

    for i, v in enumerate(final_scores['rating'].value_counts().sort_index()):
        ax4.text(v, i, str(v), color='black', fontweight='bold', fontname="Arial", fontsize=12)
        #ax.bar(i, v, edgecolor='black',bottom =-0.7, linewidth=1.5, alpha=0.8)

    one_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[1])
    two_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[2])
    three_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[3])
    four_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[4])
    five_bar = plt.bar(0, 0, color=sns.color_palette("Blues")[5])
    leg2 = plt.legend([five_bar, four_bar, three_bar,two_bar,one_bar], ['5 Stars', '4 Stars', '3 Stars', '2 Stars', '1 Star'],loc='lower right')
    for text in leg2.get_texts():
        text.set_alpha(0.5)
    leg2.legendPatch.set_alpha(0.5)
    ax4.axhline(final_scores['rating'].value_counts().mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.tight_layout()

    graph4 = 'count_of_reviews_star_rating.png'
    file_path = os.path.join(path_static, graph4)
    plt.savefig(file_path)


    print(product_name)
    print(total_reviews)
    print(total_positive)
    print(average_rating)
    
    return render_template('result.html', product_name=product_name,total_reviews=total_reviews,total_positive=total_positive,average_rating=average_rating)
    
class CleanCache:
	'''
	this class is responsible to clear any residual csv and image files
	present due to the past searches made.
	'''
	def __init__(self, directory=None):
		self.clean_path = directory
		# only proceed if directory is not empty
		if os.listdir(self.clean_path) != list():
			# iterate over the files and remove each file
			files = os.listdir(self.clean_path)
			for fileName in files:
				print(fileName)
				os.remove(os.path.join(self.clean_path,fileName))
		print("cleaned!")

if __name__ == "__main__":
    app.run(debug=True)