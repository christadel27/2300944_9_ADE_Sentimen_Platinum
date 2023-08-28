import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Membaca file CSV
df = pd.read_csv('csv_data/train_preprocess.tsv.txt', sep='\t', names=["text", "label"])

# Filter data berdasarkan sentimen
sentiments = ['negative', 'positive', 'neutral']

for sentiment in sentiments:
    sentiment_data = df[df['label'] == sentiment]
    sentiment_text = ' '.join(sentiment_data['text'])
    
    # Buat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(sentiment_text)
    
    # Tampilkan WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {sentiment} Sentiment')
    plt.show()