import tkinter as tk
#from recommendations import get_popular_recommendation, get_personalized_recommendation

def start_gui():
    window = tk.Tk()
    window.title("Film Öneri Sistemi")

    # Çıktı ekranı
    output_label = tk.Label(window, text="Önerilen film burada gösterilecek")
    output_label.pack()

    # Öneri tipi seçim alanı
    option_type = tk.StringVar(value="populer")  # Varsayılan olarak popüler öneriler
    popular_button = tk.Radiobutton(window, text="Popüler Film Önerileri", variable=option_type, value="populer")
    personalized_button = tk.Radiobutton(window, text="Kişiselleştirilmiş Film Önerileri", variable=option_type, value="personalized")
    popular_button.pack()
    personalized_button.pack()

    # Öneri yap butonu
    def get_recommendation():
        if option_type.get() == "populer":
            recommendation = get_popular_recommendation()
        else:
            recommendation = get_personalized_recommendation()
        output_label.config(text=f"Önerilen Film: {recommendation}")

    recommend_button = tk.Button(window, text="Film Öner", command=get_recommendation)
    recommend_button.pack()

    window.mainloop()