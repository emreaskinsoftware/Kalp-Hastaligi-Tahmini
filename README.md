# Kalp-Hastaligi-Tahmini
Python ve Scikit-learn ile Makine Öğrenimi projesi: Kalp hastalığı tahmini
# Makine Öğrenimi Projesi: Kalp Hastalığı Tahmini

**Amaç:** Bu proje, bir hastanın tıbbi verilerine (yaş, kolesterol, kan basıncı vb.) dayanarak, o kişinin kalp hastası olup olmadığını (%88'in üzerinde bir doğrulukla) tahmin eden bir Makine Öğrenimi (Sınıflandırma) modeli geliştirmeyi amaçlamaktadır.

**Portföydeki Etkisi:** Bu proje, baştan sona bir veri bilimi yaşam döngüsünü uygulama becerimi göstermektedir:
* Ham ve kirli veriyi analiz etme ve temizleme.
* Modelin anlayacağı formata getirmek için **Ön İşleme (Preprocessing)** ve **Özellik Mühendisliği (Feature Engineering)** yapma.
* **Scikit-learn (sklearn)** kütüphanesi ile model eğitme ve değerlendirme.
* Modeli **iyileştirme (Optimization)** (örn: Feature Scaling) ve farklı modelleri (Lojistik Regresyon vs. Random Forest) karşılaştırma.

**Kullanılan Araçlar:**
* Python
* Pandas (Veri temizleme, filtreleme ve ön işleme için)
* Scikit-learn (StandardScaler, LogisticRegression, RandomForestClassifier, train_test_split, accuracy_score)
* Matplotlib & Seaborn (İlk analizler için)
* Google Colab (Analiz ortamı)

---

## 🧭 Analiz ve Modelleme İş Akışı (Workflow)

Proje, "ham" veriden "tahmin" modeline giden 4 ana adımda tamamlanmıştır:

### 1. Veri Keşfi ve Temizleme

* **İlk Analiz:** Yüklenen ham veri seti (920 hasta), aslında 4 farklı tıbbi merkezden (Cleveland, Hungary, vb.) toplanan verilerin birleştirilmiş haliydi.
* **Kritik Tespit:** Diğer 3 merkezdeki verilerde `ca`, `slope` ve `thal` gibi kritik sütunlarda %50'nin üzerinde eksik veri olduğu tespit edildi.
* **Profesyonel Karar:** Modeli yanıltıcı verilerle eğitmek yerine, bu 4 set içindeki **en eksiksiz ve en güvenilir** alt küme olan **"Cleveland"** veri seti (304 hasta) ile çalışmaya karar verildi.

### 2. Veri Ön İşleme (Preprocessing)

Modelimizi eğitebilmek için "Cleveland" alt seti üzerinde iki temel dönüşüm yapıldı:

1.  **Eksik Veri Yönetimi:** Yeni setteki az sayıdaki (%3'ten az) eksik veri içeren satırlar, veri bütünlüğünü bozmadığı için `dropna()` ile temizlendi (Son veri boyutu: 297 hasta).
2.  **Özellik Mühendisliği (Feature Engineering):**
    * **Hedef Değişken (`target`):** Orijinal `num` (0-4 arası) sütunu, modelimizin amacı olan "ikili sınıflandırma" (binary classification) için `0` (Sağlıklı) ve `1` (Hasta) olacak şekilde yeniden kodlandı.
    * **Kategorik Veri:** Modelin anlayabilmesi için `sex` (Male/Female), `cp` (angina types) gibi tüm `object` (metin) tipindeki sütunlar, `pd.get_dummies()` (One-Hot Encoding) yöntemiyle sayısal formata (0/1) dönüştürüldü.

### 3. Baseline Model (Temel Model)

* Veri seti %80 Eğitim (Train) ve %20 Test olarak ayrıldı.
* İlk temel model olarak `LogisticRegression` kullanıldı.
* **İlk Sonuç (Baseline): %86.67 Doğruluk (Accuracy)**
* **Tespit Edilen Sorun:** Model eğitilirken, `age` (30-70), `chol` (120-500) ve `sex` (0-1) gibi sütunlar arasındaki devasa ölçek farkları nedeniyle bir `ConvergenceWarning` (Yakınsama Uyarısı) alındı.

### 4. Model İyileştirme ve Değerlendirme

Modeli hem profesyonel standartlara getirmek (uyarıyı gidermek) hem de başarısını artırmak için **Özellik Ölçeklendirme (Feature Scaling)** uygulandı:

1.  **Ölçeklendirme:** `StandardScaler` kullanılarak tüm `X_train` ve `X_test` verilerinin ölçeği (ortalama 0, std 1) eşitlendi.
2.  **Modelin Yeniden Eğitilmesi:** Ölçeklenmiş veri ile `LogisticRegression` modeli *tekrar* eğitildi.

---

## 📊 Sonuçlar ve Karşılaştırma

#### Kazanç 1: Teknik Başarı
`StandardScaler` kullanıldıktan sonra `ConvergenceWarning` uyarısı **başarıyla giderildi**. Bu, modelin artık matematiksel olarak daha stabil ve güvenilir bir çözüm bulduğunu kanıtladı.

#### Kazanç 2: Model Başarısı
Ölçeklendirme, modelin başarısını doğrudan artırdı ve **özellikle "Hasta" (`1`) sınıfını yakalama başarısını (`f1-score`) yükseltti.**

| Model | Doğruluk (Accuracy) | 'Hasta' Sınıfı f1-score |
| :--- | :---: | :---: |
| Baseline (Ölçeklenmemiş LR) | 86.67% | 0.83 |
| **İyileştirilmiş (Ölçeklenmiş LR)** | **88.33%** | **0.86** |

#### Deney 3: Model Karşılaştırması
Daha karmaşık bir model olan `RandomForestClassifier` da denendi. Bu model de **%88.33** doğruluk skoru verdi.

**Proje Sonucu (Insight):** Her iki modelin de aynı sonucu vermesi, verimizdeki "hasta" ve "sağlıklı" ayrımının `LogisticRegression` gibi daha basit (lineer) bir modelle bile etkili bir şekilde yakalanabildiğini göstermektedir. Bu durumda, daha hızlı ve yorumlanması daha kolay olan **Lojistik Regresyon modeli** tercih edilen modeldir.
