import tkinter as tk
import pandas as pd
from app.apriori import AprioriRecommender


# Kullanıcı verilerini yükleyelim
def load_users(file_path):
    df = pd.read_csv('data/ratings.csv')
    user_ids = df['userId'].tolist()
    return user_ids, df


# Apriori algoritmasını çalıştırarak öneri alalım
def apriori_recommendation(df, user_id):
    # Kullanıcıların izlediği filmleri filtreleyelim
    user_movies = df[df['userId'] == user_id]['movieId'].tolist()

    # Apriori algoritmasını çalıştırmak için uygun formatta veri hazırlayalım
    transactions = df.groupby('userId')['movieId'].apply(list).tolist()

    # Apriori algoritması
    rules = AprioriRecommender(transactions, min_support=0.01, min_confidence=0.5, min_lift=1.0)

    # Öneriler için sonuçları işleyelim
    recommendations = []
    for rule in rules:
        for item in rule.items:
            if item[0] in user_movies:
                recommendations.extend(rule.items)

    # Tekrar eden film ID'leri kaldırıp önerileri döndürelim
    return list(set(recommendations))


# Tkinter GUI uygulaması
class MovieRecommendationApp:
    def __init__(self, root, user_ids, df):
        self.root = root
        self.root.title("Film Öneri Sistemi")
        self.df = df

        # Kullanıcıları bir Listbox'a yerleştirelim
        self.user_listbox = tk.Listbox(self.root)
        self.user_listbox.pack(pady=20)

        # Kullanıcı ID'lerini Listbox'a ekleyelim
        for user_id in user_ids:
            self.user_listbox.insert(tk.END, user_id)

        # Kullanıcı seçim işlemi
        self.select_button = tk.Button(self.root, text="Kullanıcı Seç", command=self.select_user)
        self.select_button.pack(pady=10)

        # Sonuçların gösterileceği çıktı ekranı
        self.result_label = tk.Label(self.root, text="Önerilen Filmler:", justify="left")
        self.result_label.pack(pady=10)

        # Önerilen filmleri gösterecek alan
        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.pack(pady=10)

    def select_user(self):
        selected_user_id = self.user_listbox.get(tk.ACTIVE)  # Seçilen ID'yi al
        print(f"Seçilen Kullanıcı ID: {selected_user_id}")

        # Apriori ile öneri alalım
        recommendations = apriori_recommendation(self.df, selected_user_id)

        # Önerilen filmleri çıktı ekranında gösterelim
        self.output_text.delete(1.0, tk.END)  # Mevcut metni temizleyelim
        if recommendations:
            for movie in recommendations:
                self.output_text.insert(tk.END, f"Film ID: {movie}\n")
        else:
            self.output_text.insert(tk.END, "Öneri bulunamadı.\n")


# Ana program çalıştırma
if __name__ == "__main__":
    # Kullanıcı verilerini yükle
    user_ids, df = load_users('user_data.csv')  # user_data.csv dosyasını kendi dosyanızla değiştirin

    # Tkinter penceresini oluştur
    root = tk.Tk()
    app = MovieRecommendationApp(root, user_ids, df)
    root.mainloop()
