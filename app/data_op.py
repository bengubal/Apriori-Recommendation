import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend as mlx
from pandas import merge
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
#from pyspark.sql import SparkSession
#from pyspark.sql.functions import col, when
#from pyspark.mllib.linalg import SparseVector
from app.main import adventure_average_ratings, animation_average_ratings, children_average_ratings, \
    comedy_average_ratings, drama_average_ratings, fantasy_average_ratings, horror_average_ratings, \
    romance_average_ratings, sci_fi_average_ratings, thriller_average_ratings, war_average_ratings

horror_average_ratings, romance_average_ratings,sci_fi_average_ratings, thriller_average_ratings, war_average_ratings

popular_genre = adventure_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/adventure_cleaned.csv',index=False)

popular_genre = animation_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/animation_cleaned.csv',index=False)

popular_genre = children_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/children_cleaned.csv',index=False)

popular_genre = comedy_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/comedy_cleaned.csv',index=False)

popular_genre = drama_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/drama_cleaned.csv',index=False)

popular_genre = fantasy_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/fantasy_cleaned.csv',index=False)

popular_genre = horror_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/horror_cleaned.csv',index=False)

popular_genre = romance_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/romance_cleaned.csv',index=False)

popular_genre = sci_fi_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/sci_fi_cleaned.csv',index=False)

popular_genre = thriller_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/thriller_cleaned.csv',index=False)

popular_genre = war_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
popular_genre.to_csv('data/war_cleaned.csv',index=False)

folder_path = "popular-csv"  # Klasör yolunu buraya yazın
output_file = "popular_movies.csv"  # Çıkış dosyasının adı

# Tüm CSV dosyalarını bir listede birleştir
dataframes = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Tüm DataFrame'leri tek bir DataFrame'de birleştir
merged_df = pd.concat(dataframes, ignore_index=True)

# Tek bir CSV dosyasına kaydet
merged_df.to_csv(output_file, index=False)
print(f"Bütün dosyalar '{output_file}' ismiyle birleştirildi.")




# veri setini yüklüyorum
file_path = 'data/rating_cleaned.csv'
data = pd.read_csv(file_path)

# userId ve movieId değerlerini alıyorum
unique_users = data['userId'].unique()
unique_movies = data['movieId'].unique()

# yukarıda aldığım değerleri indexlere dönüştürüyorum
user_index = {user: idx for idx, user in enumerate(unique_users)}
movie_index = {movie: idx for idx, movie in enumerate(unique_movies)}

# satırlara user sütunlara movie değeri gelecek şekilde atama yapıyorum
rows = [user_index[user] for user in data['userId']]
cols = [movie_index[movie] for movie in data['movieId']]
data_values = [1] * len(data)  # Tüm rating olan değerlere 1 atıyoruz

# matrisi oluşturuyorum
user_movie_matrix = csr_matrix((data_values, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

# oluşturulan matriste eksik veri var mı görmek için user ve movie sayılarını kontrol ediyorum
print("Kullanıcı sayısı:", len(unique_users))
print("Film sayısı:", len(unique_movies))
print("Kullanıcı-Film matrisinin şekli:", user_movie_matrix.shape)

print(user_movie_matrix.toarray()[:10, :10])
print(user_movie_matrix)
