import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend as mlx
import self
from pandas import merge
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from app.apriori import AprioriRecommender
from app.gui import MovieRecommendationApp





#veriyi tek satırda görebilmek için yaptırdığımız işlemler
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


movies_df = pd.read_csv('data/movies_cleaned.csv', index_col=0)
print(movies_df.head(5))
print("---------------------------------------------")
ratings_df = pd.read_csv('data/rating_cleaned.csv', index_col=0)
print(ratings_df.head(5))

popular_movies = pd.read_csv('popular_movies.csv')
movies_cleaned = pd.read_csv('data/movies_cleaned.csv')

# Title değerlerine göre birleştirme
merged_df = pd.merge(popular_movies, movies_cleaned[['title', 'genres']], on='title', how='left')

# genre sütununu popüler filmlere ekleyin
popular_movies['genre_from_cleaned'] = merged_df['genres']

# Sonuçları kontrol edin
print(popular_movies.head())


#film ve rating veri setlerini birlesitiryorum
merge_df = merge(movies_df, ratings_df, on='movieId')
#merge_df.to_csv('data/merged.csv')

#çıktı alırken karışıklık olmaması için index sütunu kaldırmak için
print(merge_df.tail(20).to_string(index=False))

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#veri setini turlere gora ayirmak icin yaptirdigim islemler
genres_df = movies_df[['movieId', 'title', 'genres']].copy()
genres_df['genres'] = genres_df['genres'].str.split('|')
genres_df=genres_df.explode('genres')
genres_df.sort_values(by=['genres'], ascending=True, inplace=True)



print("-----------------------------------")



print("kullanıcı matrisi:___________________________")

# veri setini yüklüyorum
file_path = 'data/ratings.csv'
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




movies_cleaned_path = 'data/movies_cleaned.csv'  # Bu yolu güncellediğinizden emin olun
popular_movies_path = 'all_popular.csv'  # Popüler filmler için dosya yolu

# # Önerici sınıfını başlat
# recommender = AprioriRecommender(user_movie_matrix, popular_movies_path, movies_cleaned_path)
#
# # Modeli eğitin
# recommender.fit()
#
recommender = AprioriRecommender(user_movie_matrix, popular_movies_path, movies_cleaned_path)
recommender.fit()
# # Kullanıcının izlediği film ID'lerini alın
user_id = 59  # Geçerli bir kullanıcı ID'si
user_movie_ids = user_movie_matrix[user_id].nonzero()[1].tolist()  # İzlenen filmlerin indekslerini alıyoruz

# Kullanıcının izlediği film başlıklarını almak için ID'leri kullanın
user_movies = [recommender.get_movie_title(movie_id) for movie_id in user_movie_ids]

# Film önerisi almak
recommendations = recommender.recommend(user_id)
print("Kullanıcıya önerilen filmler:", [recommender.get_movie_title(movie_id) for movie_id in recommendations])

recommendbymovie = recommender.recommend_for_movie(234)
print("filme göre önerilen filmler:", recommendbymovie)

# Türüne göre öneri almak
genre = 'Horror'  # Örnek tür
genre_recommendations = recommender.recommend_by_genre(genre, user_movies)
print("Türe göre öneriler:", genre_recommendations)

MovieRecommendationApp()


