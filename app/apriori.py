import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix


class AprioriRecommender:
    def __init__(self, user_movie_matrix, popular_movies_path, movies_cleaned_path, min_support=0.15, min_confidence=0.6):
        # Sparse matrisi dense matrise çevir
        self.df = pd.DataFrame(user_movie_matrix.toarray(), columns=range(user_movie_matrix.shape[1])).astype(bool)
        self.popular_movies = pd.read_csv(popular_movies_path)
        self.movies_cleaned = pd.read_csv(movies_cleaned_path)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.rules = None

    def fit(self):
        """Apriori algoritmasını çalıştır ve sık öğe kümelerini bul."""
        # Apriori ile sık öğe kümelerini belirle
        self.frequent_itemsets = apriori(self.df, min_support=self.min_support, use_colnames=True)
        # Birliktelik kurallarını oluştur
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)

        # Destek, güven ve lift değerlerini hesapla
        self.rules['support'] = self.rules['support'].round(4)
        self.rules['confidence'] = self.rules['confidence'].round(4)
        self.rules['lift'] = self.rules['lift'].round(4)

    def get_movie_title(self, movie_id):
        """Film ID'sinden başlık döndür."""
        title = self.movies_cleaned.loc[self.movies_cleaned['movieId'] == movie_id, 'title']
        return title.iloc[0] if not title.empty else None

    def recommend(self, user_id):
        """Belirli bir kullanıcı için film önerilerini döndürür."""
        if self.rules is None:
            raise ValueError("Model henüz eğitilmedi. Önce 'fit()' fonksiyonunu çalıştırın.")

        # Kullanıcının izlediği filmleri al
        user_movies = set(self.df.columns[self.df.iloc[user_id] == True])
        print(f"Kullanıcı {user_id} tarafından izlenen filmler:", user_movies)

        # Kullanıcının izlediği filmlerle ilişkili kuralları filtrele
        relevant_rules = self.rules[self.rules['antecedents'].apply(lambda x: x.issubset(user_movies))]

        if relevant_rules.empty:
            print(f"Kullanıcı {user_id} için geçerli kural bulunamadı.")
            return []

        recommendations = set()
        for _, row in relevant_rules.iterrows():
            # Kullanıcının izlemediği film önerilerini ekle
            consequents = row['consequents'] - user_movies
            recommendations.update(consequents)

        return list(recommendations)

    def recommend_by_title(self, movie_title, user_movies, rules):
        """Film ismine göre öneriler döndürür."""
        if movie_title not in self.popular_movies['title'].values:
            print(f"{movie_title} için öneri bulunamadı.")
            return []

        movie_id = self.popular_movies.loc[self.popular_movies['title'] == movie_title, 'movieId'].values[0]

        # Filmle ilişkili olan diğer filmleri öner
        recommendations = self.recommend_movies_by_movie(movie_id, rules)

        # Kullanıcının izlediği filmleri önerilerden çıkar
        user_movie_ids = [self.popular_movies.loc[self.popular_movies['title'] == title, 'movieId'].values[0] for title
                          in user_movies if title in self.popular_movies['title'].values]
        filtered_recommendations = [movie for movie in recommendations if movie not in user_movie_ids]

        # Film başlıklarını geri döndür
        return [self.popular_movies.loc[self.popular_movies['movieId'] == rec, 'title'].values[0] for rec in
                filtered_recommendations]


    def recommend_by_genre(self, genre, user_movies):
        """Türe göre öneriler döndürür."""
        relevant_movies = self.popular_movies[self.popular_movies['genre'].str.contains(genre)]

        # Kullanıcının izlediği filmlerden önerileri filtrele popüler
        user_movie_ids = [self.popular_movies.loc[self.popular_movies['title'] == title, 'movieId'].values[0] for title in user_movies if title in self.popular_movies['title'].values]
        recommended_movies = relevant_movies[~relevant_movies['movieId'].isin(user_movie_ids)]

        return list(recommended_movies['title'])

    def recommend_for_movie(self, movie_id):
        """Belirli bir film için öneri döndürür."""
        if self.rules is None:
            raise ValueError("Model henüz eğitilmedi. Önce 'fit()' fonksiyonunu çalıştırın.")

        # Film için ilişkili kuralları filtrele
        relevant_rules = self.rules[self.rules['antecedents'].apply(lambda x: movie_id in x)]
        if relevant_rules.empty:
            print(f"{movie_id} ID'li film için geçerli kural bulunamadı.")
            return []

        # Filmin ilişkili olduğu diğer filmleri öner
        recommendations = set()
        for _, row in relevant_rules.iterrows():
            consequents = row['consequents']
            recommendations.update(consequents)

        # Film ID'lerini başlıklara dönüştür
        movie_titles = [self.get_movie_title(movie_id) for movie_id in recommendations]
        return movie_titles