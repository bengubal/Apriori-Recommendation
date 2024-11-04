import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix

class AprioriRecommender:
    def __init__(self, user_movie_matrix, min_support=0.3, min_confidence=0.3):
        # Sparse matrisi dense matrise çevir
        self.df = pd.DataFrame(user_movie_matrix.toarray(), columns=range(user_movie_matrix.shape[1])).astype(bool)
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
        else:
            print(f"Kullanıcı {user_id} için geçerli kurallar:", relevant_rules)

        recommendations = set()
        for _, row in relevant_rules.iterrows():
            # Kullanıcının izlemediği film önerilerini ekle
            consequents = row['consequents'] - user_movies
            recommendations.update(consequents)

        return list(recommendations)


