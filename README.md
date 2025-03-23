# News--Category-Dataset
# 📰 Haber Kategorisi Sınıflandırma Projesi

Bu proje, haber metinlerini otomatik olarak sınıflandırmak için makine öğrenimi tekniklerini kullanmaktadır. Farklı haber kategorilerinin (örneğin, iş, eğlence, sağlık) belirlenmesi amacıyla metin verisi işlenmekte ve derin öğrenme modellemesi yapılmaktadır.

⚠️ Not
3D grafiklerim ve görselleştirmelerim maalesef gözükmüyor. Bu durum, bazı tarayıcı veya platform uyumsuzluklarından kaynaklanabilir.

## 🔗 Kaggle Veri Seti
[Haber Kategorisi Veri Seti](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

## 🔗 Hugging Face Uygulaması
[Haber Kategorisi Veri Seti - Hugging Face Space] sorun yaşadım

## 🔗 Uygulama Linki
Uygulamayı yerel olarak çalıştırmak için: [http://localhost:8501/](http://localhost:8501/)

## 📊 Proje Aşamaları
1. **Veri Yükleme**:
   - `News_Category_Dataset_v3.json` dosyası yüklenir ve DataFrame'e dönüştürülür.

2. **Metin Temizleme**:
   - Başlık ve kısa açıklama birleştirilir.
   - Metin, küçük harfe dönüştürülür, noktalama işaretleri kaldırılır, stopword'ler filtrelenir ve lemmatization uygulanır.

3. **Veri Analizi**:
   - Kategori dağılımı için görselleştirme yapılır.

4. **Veriyi Eğitim ve Test Olarak Ayırma**:
   - Temizlenmiş metinler eğitim ve test setlerine ayrılır.

5. **Metin Tokenizasyonu ve Doldurma**:
   - Metin verileri, Tokenizer kullanılarak sayısal verilere dönüştürülür ve belirli bir uzunluğa pad edilir.

6. **Model Oluşturma**:
   - LSTM tabanlı bir model oluşturulur ve derin öğrenme ile eğitilir.

7. **Model Eğitimi**:
   - Modelin eğitimi gerçekleştirilir ve doğruluk ile kayıp değerleri görselleştirilir.

8. **Test Verisi Üzerinde Tahmin**:
   - Model test seti üzerinde tahmin yapar ve doğruluk skoru hesaplanır.

9. **Sonuç Analizi**:
   - Sınıflandırma raporu ve karmaşıklık matrisleri görselleştirilir.

10. **Örnek Tahminler**:
    - Modelin tahmin yeteneği, örnek metinler üzerinde gösterilir.

11. **Modeli Kaydetme**:
    - Eğitilen model, Tokenizer ve LabelEncoder dosyaları kaydedilir.

## 📈 Model Performansı
- Test doğruluğu: `0.8043`
- Kategoriler için sınıflandırma raporu ve karmaşıklık matrisleri sunulacaktır.

## 🛠️ Kullanılan Kütüphaneler
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk` (metin işleme için)
- `tensorflow` (derin öğrenme için)

Bu proje, haber metinlerini sınıflandırarak kullanıcıların belirli konular hakkında daha hızlı bilgi edinmelerini sağlamayı hedeflemektedir. Elde edilen model, çeşitli haber kategorilerinin otomatik olarak belirlenmesine yardımcı olmaktadır.
