import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Veri seti (Cümleler ve etiketleri)
data = [
    ("Bu ürün çok güzel", 1),  # 1: İyi
    ("Harika bir film izledim", 1),  # 1: İyi
    ("Bugün çok kötü bir gün", 0),  # 0: Kötü
    ("Mükemmel bir yemek yedim", 1),  # 1: İyi
    ("Kötü bir alışveriş deneyimi", 0)  # 0: Kötü
]

# Cümleleri ve etiketleri ayıralım
texts = [text for text, label in data]
labels = [label for text, label in data]

# Kelimeleri sayılara dönüştürmek için Tokenizer kullanıyoruz
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Veriyi sayılara dönüştürme
sequences = tokenizer.texts_to_sequences(texts)

# Veriyi sabit uzunlukta olacak şekilde pad'liyoruz
max_length = 10  # Cümle uzunluğu
X = pad_sequences(sequences, maxlen=max_length)

# Etiketler
y = np.array(labels)

# Modeli oluşturma
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))  # Çıktı: 0 veya 1

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X, y, epochs=10, batch_size=2)

# Yeni veri (modeli test etmek için)
new_texts = ["Bu film çok kötüydü"]

# Yeni veriyi sayılara dönüştürme
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=max_length)

# Tahmin yapma
prediction = model.predict(new_X)
print(f"Tahmin: {'Kötü' if prediction[0] < 0.5 else 'İyi'}")

# Ana çalışma bloğu
if __name__ == "__main__":
    print("Yapay zeka scripti başarıyla çalıştı!")
