import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import sys
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# App title and description
st.set_page_config(page_title="Haber Kategorisi Sınıflandırıcı", layout="wide")
st.title("Haber Kategorisi Sınıflandırma Uygulaması")
st.markdown("Bu uygulama, verilen haber metinlerini kategorilere ayırır.")

# NLTK veri dizinini ayarla ve gerekli paketleri indir
try:
    import nltk
    nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
    
    # Download NLTK resources if not already downloaded
    @st.cache_resource
    def download_nltk_data():
        for resource in ["punkt", "stopwords", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
                st.write(f"{resource} zaten indirilmiş.")
            except LookupError:
                st.write(f"{resource} indiriliyor...")
                nltk.download(resource)
    
    download_nltk_data()
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    nltk_loaded = True
    st.success("NLTK başarıyla yüklendi!")
except Exception as e:
    st.error(f"NLTK yüklenirken hata oluştu: {e}")
    nltk_loaded = False

# Load the model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('news_category_classifier.h5')
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
            
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return model, tokenizer, label_encoder, pad_sequences
    except Exception as e:
        st.error(f"Model yükleme hatası: {str(e)}")
        return None, None, None, None

if nltk_loaded:
    model, tokenizer, label_encoder, pad_sequences = load_model()
    if model is not None:
        st.success("Model ve gerekli bileşenler başarıyla yüklendi!")
        model_loaded = True
    else:
        model_loaded = False
else:
    model_loaded = False

# Text cleaning function
def clean_text(text):
    if not nltk_loaded:
        return text
    
    # Küçük harfe dönüştürme
    text = text.lower()
    # Noktalama işaretlerini kaldırma
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Fazladan boşlukları kaldırma
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize etme
    tokens = word_tokenize(text)
    # Stopword'leri kaldırma
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Prediction function
def predict_category(text, max_len=200):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)[0]
    pred_class = label_encoder.classes_[np.argmax(pred)]
    
    # Get top 3 predictions with probabilities
    top_indices = pred.argsort()[-3:][::-1]
    top_classes = [label_encoder.classes_[i] for i in top_indices]
    top_probs = [pred[i] for i in top_indices]
    
    return pred_class, top_classes, top_probs

# Main app functionality
if nltk_loaded and model_loaded:
    st.header("Haber Metni Sınıflandırma")

    # Input options
    input_option = st.radio("Giriş yöntemi seçin:", ["Tek metin girişi", "Toplu metin girişi (CSV dosyası)"])

    if input_option == "Tek metin girişi":
        # Single text input
        headline = st.text_input("Haber Başlığı:")
        description = st.text_area("Haber Açıklaması:")
        
        if st.button("Sınıflandır") and (headline or description):
            with st.spinner("Sınıflandırılıyor..."):
                text = headline + " " + description
                pred_class, top_classes, top_probs = predict_category(text)
                
                # Display results
                st.subheader("Sınıflandırma Sonuçları")
                st.markdown(f"**Tahmin Edilen Kategori:** {pred_class}")
                
                # Create a bar chart for top predictions
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                bars = ax.bar(top_classes, [prob * 100 for prob in top_probs], color=colors)
                
                # Add percentage labels
                for bar, prob in zip(bars, top_probs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{prob*100:.1f}%', ha='center', va='bottom')
                
                plt.ylabel('Olasılık (%)')
                plt.title('En Olası Kategoriler')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display cleaned text
                st.subheader("Metin Ön İşleme")
                st.text_area("Temizlenmiş Metin:", value=clean_text(text), height=100, disabled=True)

    else:
        # Batch processing with CSV upload
        st.subheader("Toplu Sınıflandırma için CSV Dosyası Yükleyin")
        st.markdown("CSV dosyanızda **'text'** sütunu olmalıdır.")
        
        uploaded_file = st.file_uploader("CSV dosyası seçin", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV dosyasında 'text' sütunu bulunamadı.")
                else:
                    if st.button("Toplu Sınıflandırma Yap"):
                        with st.spinner("Sınıflandırılıyor... Bu işlem biraz zaman alabilir."):
                            # Process each text and store results
                            results = []
                            for idx, row in df.iterrows():
                                text = row['text']
                                pred_class, _, _ = predict_category(text)
                                results.append(pred_class)
                            
                            # Add predictions to dataframe
                            df['predicted_category'] = results
                            
                            # Display results
                            st.subheader("Sınıflandırma Sonuçları")
                            st.dataframe(df)
                            
                            # Show category distribution
                            st.subheader("Kategori Dağılımı")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            category_counts = df['predicted_category'].value_counts()
                            sns.barplot(x=category_counts.index, y=category_counts.values)
                            plt.xticks(rotation=90)
                            plt.title('Tahmin Edilen Kategori Dağılımı')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Option to download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="CSV olarak indir",
                                data=csv,
                                file_name="siniflandirma_sonuclari.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"CSV işlenirken hata oluştu: {e}")
else:
    st.error("NLTK ve model yüklenene kadar uygulama çalışamaz. Lütfen hataları giderin.")
    st.info("Sistem bilgisi:")
    st.write(f"Python sürümü: {sys.version}")
    st.write(f"Çalışma dizini: {os.getcwd()}")
    st.write("Dizin içeriği:")
    st.write(os.listdir())

# Add information about the model
st.sidebar.header("Model Bilgisi")
st.sidebar.markdown("""
Bu uygulama, haber metinlerini aşağıdaki kategorilere sınıflandırır:
- BUSINESS  
- ENTERTAINMENT
- FOOD & DRINK
- HEALTHY LIVING
- PARENTING
- POLITICS
- QUEER VOICES
- STYLE & BEAUTY
- TRAVEL
- WELLNESS
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Kullanım Talimatları")
st.sidebar.markdown("""
1. Tek metin girişi veya toplu metin girişi yöntemini seçin
2. Metin(leri) girin veya CSV dosyası yükleyin
3. Sınıflandır butonuna tıklayın
4. Sonuçları inceleyin
""")

# Footer
st.markdown("---")
st.markdown("Haber Kategorisi Sınıflandırma Projesi © 2025")
