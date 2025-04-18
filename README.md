# ApplicantSelection

# SVM Hiring Prediction API

Bu proje, deneyim yılı ve teknik test puanı gibi faktörlere dayanarak bir adayın işe alınıp alınmayacağını tahmin eden bir Makine Öğrenimi modeline dayalı bir FastAPI uygulamasıdır.

Uygulamayı çalıştırmak için:

```bash
uvicorn main:app --reload
```

## API Uç Noktaları

### 1. Ana Sayfa

**GET /**

API'nin çalıştığını doğrulamak için kullanılabilir.

**Yanıt:**

```json
{
  "message": "SVM API is running"
}
```

---

### 2. SVM Raporu

**GET /svm_report?metric=all**

**Parametreler:**

- `metric`:  
  - `accuracy`  
  - `confusion_matrix`  
  - `classification_report`  
  - `all` (tüm metrikleri döner)

**Örnek Kullanım:**

```bash
GET /svm_report?metric=accuracy
```

**Yanıt:**

```json
{
  "accuracy": 0.86
}
```

---

### 3. Tahmin (Prediction)

**POST /predict**

Aday bilgileri ile tahmin yapmak için bu endpoint'e istek gönderin.

**Request Body (JSON):**

```json
{
  "experience_years": 5,
  "technical_score": 75
}
```

**Yanıt:**

```json
{
  "experience_years": 5,
  "technical_score": 75,
  "prediction": 1
}
```

---

## Model Detayları

- **Algoritma:** Support Vector Machine (Linear Kernel)
- **Özellikler:**
  - `experience_years`
  - `technical_score`
- **Hedef:** `hire_label` (0: Alınmaz, 1: Alınır)
- Model eğitim verileri `createDataBase.py` içinde oluşturulur.
- Modelin değerlendirme metrikleri `modelTest.py` ile analiz edilir.

---

## Notlar

- Model çıktıları eğitim verisindeki gürültü oranı (%5) ile değişebilir.
- Model `.pkl` formatında kaydedilir (`modelSave.py`) ve tahminlerde kullanılır.
- Uygulama Swagger arayüzü ile test edilebilir:  
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Geliştiren

**Ece Sude** – Computer Engineering Student  
**Staj Projesi** – SVM Tabanlı İşe Alım Tahmin Sistemi

