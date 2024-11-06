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
#popular_movies.to_csv('data/popular_movies.csv')
#user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Kullanıcı-film matrisini yazdırma
#print("User-Movie Matrix:")
#print(user_movie_matrix.head(5))

# İlk 10.000 kullanıcı ile pivot tablo oluşturma
#subset_ratings_df = ratings_df[ratings_df['userId'] < 1000]
#user_movie_matrix = subset_ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Kullanıcı-film matrisini yazdırma
#print("User-Movie Matrix:")
#print(user_movie_matrix.head(5))

#movies_df= movies_df.drop('Unnamed: 0', axis=1)



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

#genres_df.to_csv('data/genres.csv')

#ayirdigim tur ve rating veri setlerini birlestiriyorum
# merge_genre_df = merge(genres_df, ratings_df, on='movieId')
# merge_genre_df.sort_values(by=['genres', 'rating'], ascending=[True, False], inplace=True)
#
# merge_genre_df.to_csv('data/genre_rating.csv')
# print("-----------------------------")

#toplu veri setini turlere gore ayiriyorum ve her tur icin ayri bir df olusturuyorum
# genre_dataframes = {
#     genre: merge_genre_df[merge_genre_df['genres'] == genre]
#     for genre in merge_genre_df['genres'].unique()
# }
#
# comedy_df=genre_dataframes['Comedy']
# action_df =genre_dataframes['Action']
# animation_df=genre_dataframes['Animation']
# romance_df=genre_dataframes['Romance']
# adventure_df=genre_dataframes['Adventure']
# drama_df=genre_dataframes['Drama']
# horror_df=genre_dataframes['Horror']
# fantasy_df=genre_dataframes['Fantasy']
# sci_fi_df=genre_dataframes['Sci-Fi']
# thriller_df=genre_dataframes['Thriller']
# war_df=genre_dataframes['War']
# children_df=genre_dataframes['Children']

##ture ozel dfleri kaydediyorum ki daha sonra islem yapabileyim
# comedy_df.to_csv('data/comedy.csv')
# action_df.to_csv('data/action.csv')
# animation_df.to_csv('data/animation.csv')
# romance_df.to_csv('data/romance.csv')
# adventure_df.to_csv('data/adventure.csv')
# drama_df.to_csv('data/drama.csv')
# horror_df.to_csv('data/horror.csv')
# fantasy_df.to_csv('data/fantasy.csv')
# sci_fi_df.to_csv('data/sci-fi.csv')
# thriller_df.to_csv('data/thriller.csv')
# war_df.to_csv('data/war.csv')
# children_df.to_csv('data/children.csv')

# # #her tur dosyasi icin her bir filmin rating degerinin ortalamasinin buluyorum ve bu duzenledigim verileri yeni bir dosyaya kaydediyorum
# action_average_ratings = action_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# action_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # action_average_ratings.to_csv('data/action_cleaned.csv')
# #
# comedy_average_ratings = comedy_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# comedy_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # comedy_average_ratings.to_csv('data/comedy_cleaned.csv')
# #
# adventure_average_ratings = adventure_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# adventure_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # adventure_average_ratings.to_csv('data/adventure_cleaned.csv')
# #
# animation_average_ratings = animation_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# animation_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # animation_average_ratings.to_csv('data/animation_cleaned.csv')
# #
# children_average_ratings = children_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# children_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
#
# # children_average_ratings.to_csv('data/children_cleaned.csv')
# #
# drama_average_ratings = drama_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# drama_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
#
# # drama_average_ratings.to_csv('data/drama_cleaned.csv')
# #
# fantasy_average_ratings = fantasy_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# fantasy_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # fantasy_average_ratings.to_csv('data/fantasy_cleaned.csv')
# #
# horror_average_ratings = horror_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# horror_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # horror_average_ratings.to_csv('data/horror_cleaned.csv')
# #
# romance_average_ratings = romance_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# romance_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # romance_average_ratings.to_csv('data/romance_cleaned.csv')
# #
# sci_fi_average_ratings = sci_fi_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# sci_fi_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # sci_fi_average_ratings.to_csv('data/sci_fi_cleaned.csv')
# #
# thriller_average_ratings = thriller_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# thriller_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # thriller_average_ratings.to_csv('data/thriller_cleaned.csv')
# #
# war_average_ratings = war_df.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()
# war_average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
# #
# # war_average_ratings.to_csv('data/war_cleaned.csv')

print("-----------------------------------")

#en popüler filmler sıralamasında sadece ilk 50 kalacak şekilde işlem yaptırıp tekrar aynı dosyaya kaydediyorum
# siralama_sonucu = action_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
#
# # #siralama_sonucu.to_csv('data/action_cleaned.csv',index=False)
#
#
# horror_average_ratings, romance_average_ratings,sci_fi_average_ratings, thriller_average_ratings, war_average_ratings
#
# popular_genre = adventure_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/adventure_cleaned.csv',index=False)
#
# popular_genre = animation_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/animation_cleaned.csv',index=False)
#
# popular_genre = children_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/children_cleaned.csv',index=False)
#
# popular_genre = comedy_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/comedy_cleaned.csv',index=False)
#
# popular_genre = drama_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/drama_cleaned.csv',index=False)
#
# popular_genre = fantasy_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/fantasy_cleaned.csv',index=False)
#
# popular_genre = horror_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/horror_cleaned.csv',index=False)
#
# popular_genre = romance_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/romance_cleaned.csv',index=False)
#
# popular_genre = sci_fi_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/sci_fi_cleaned.csv',index=False)
#
# popular_genre = thriller_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/thriller_cleaned.csv',index=False)
#
# popular_genre = war_average_ratings.sort_values(by='average_rating', ascending=False).head(50)
# popular_genre.to_csv('data/war_cleaned.csv',index=False)


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

# smaller_matrix = user_movie_matrix[:1000, :1000]
# # AprioriRecommender sınıfını oluşturun
# recommender = AprioriRecommender(user_movie_matrix)
# recommender.fit()
# # Öneri almak için bir kullanıcı kimliği belirleyin
# user_id = 17  # Örnek kullanıcı kimliği
# suggestions = recommender.recommend(user_id)
# # Sonuçları yazdır
# print("Önerilen Filmler:", suggestions)
#
# films_df = pd.read_csv('data/movies_cleaned.csv')  # film_id, film_adı, tür
#
# # Önerilen film ID'lerini kullanarak film adlarını ve türlerini al
# suggested_films = films_df[films_df['movieId'].isin(suggestions)]
#
# # Sonuçları yazdır
# print("Önerilen Filmler:")
# for index, row in suggested_films.iterrows():
#     print(f"Film ID: {row['movieId']}, Film Adı: {row['title']}, Tür: {row['genres']}")



# recommender = AprioriRecommender(user_movie_matrix,'all_popular.csv','data/movies_cleaned.csv')
#
# recommender.fit()  # Modeli eğit
#
# user_movies = ['Inception', 'The Matrix']  # Kullanıcının izlediği filmler
# movie_title = 'The Matrix'  # Öneri almak istediğiniz film
# genre = 'Action'  # Öneri almak istediğiniz tür
#
# # Film ismine göre öneri
# title_recommendations = recommender.recommend_by_title(movie_title, user_movies)
# print("İsme göre öneriler:", title_recommendations)
#
# # Türüne göre öneri
# genre_recommendations = recommender.recommend_by_genre(genre, user_movies)
# print("Türe göre öneriler:", genre_recommendations)
#
#
# movies_cleaned = pd.read_csv('data/movies_cleaned.csv')  # Bu yolun doğru olduğundan emin olun
#
# # movieId ile başlık arasında bir eşleme oluşturun
# movie_id_to_title = pd.Series(movies_cleaned['title'].values, index=movies_cleaned['movieId']).to_dict()
#
# # Önerici sınıfını başlat
# recommender = AprioriRecommender(user_movie_matrix, 'all_popular.csv', 'data/movies_cleaned.csv')
#
# # Modeli eğitin
# recommender.fit()
#
# # Öneri almak için kullanıcı ID'si
# user_id = 56 # Geçerli bir kullanıcı ID'si
# # Kullanıcının izlediği film ID'lerini alın
# user_movies_ids = user_movie_matrix[user_id].nonzero()[1]  # İzlenen filmlerin indekslerini alıyoruz
#
# # movieId'leri başlıklara çevirin
# user_movies = [movie_id_to_title[unique_movies[movie_id]] for movie_id in user_movies_ids]
#
# # Öneri almak istediğiniz film başlığı ve türü
# movie_title = 'The Matrix'  # Örnek film başlığı
# genre = 'Action'  # Örnek tür
#
# # Film ismine göre öneri
# title_recommendations = recommender.recommend_by_title(movie_title, user_movies)
# print("İsme göre öneriler:", title_recommendations)
#
# # Türüne göre öneri
# genre_recommendations = recommender.recommend_by_genre(genre, user_movies)
# print("Türe göre öneriler:", genre_recommendations)


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


