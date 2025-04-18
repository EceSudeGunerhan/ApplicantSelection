# SVM Tabanlı İşe Alım Tahmin Sistemi

Bu proje, adayların iş görüşmelerinde işe alınıp alınmayacağını tahmin etmek için Support Vector Machine (SVM) algoritmasını kullanır. Adayın deneyim yılı ve teknik sınav puanı temel alınarak karar verilir.

## İçerik

- `createDataBase.py`: Eğitim verilerini oluşturan dosya.
- `modelTest.py`: Modeli eğitip test eden, metrikleri döndüren dosya.
- `modelSave.py`: Eğitilen modeli kaydeden ve yeni tahminler yapmaya yarayan dosya.
- `main.py`: FastAPI ile yazılmış REST API servisidir.

## Uygulamayı çalıştırmak için:

```bash
uvicorn main:app --reload
```

## API Uç Noktaları

### 1. Ana Sayfa

**GET /**

API'nin çalıştığını doğrulamak için kullanılabilir.

Yanıt:
```json
{"message": "SVM API is running"}
```

### 2. SVM Raporu

**GET /svm_report?metric=all**

Parametreler:
- `metric`: 
  - `accuracy`
  - `confusion_matrix`
  - `classification_report`
  - `all` (tüm metrikleri döner)

Örnek Kullanım:
```bash
GET /svm_report?metric=accuracy
```

Yanıt:
```json
{
  "accuracy": 0.86
}
```

### 3. Tahmin (Prediction)

**POST /predict**

Aday bilgileri ile tahmin yapmak için bu endpoint'e istek gönderin.

Request Body (JSON):
```json
{
  "experience_years": 5,
  "technical_score": 75
}
```

Yanıt:
```json
{
  "experience_years": 5,
  "technical_score": 75,
  "prediction": 1
}
```

## Model Detayları

- **Algoritma**: Support Vector Machine (Linear Kernel)
- **Özellikler**:
  - `experience_years`
  - `technical_score`
- **Hedef**: `hire_label` (0: Alınmaz, 1: Alınır)
- Model eğitim verileri `createDataBase.py` içinde oluşturulur.
- Modelin değerlendirme metrikleri `modelTest.py` ile analiz edilir.

## Karar Sınırı Görselleştirmesi

Aşağıda SVM modelinin karar sınırı gösterilmiştir:

![Figure_1](https://github.com/user-attachments/assets/57acd396-32a7-43ba-bf3c-5ef29e3fa306)


> Not: Bu görsel `modelTest.py` içerisindeki `plot_decision_boundary` fonksiyonu ile oluşturulmuştur. 

## Notlar

- Model çıktıları eğitim verisindeki gürültü oranı (%5) ile değişebilir.
- Model `.pkl` formatında kaydedilir (`modelSave.py`) ve tahminlerde kullanılır.
- Uygulama Swagger arayüzü ile test edilebilir: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Geliştiren

**Ece Sude GÜNERHAN** – Computer Engineering Student  
**Staj Projesi** – SVM Tabanlı İşe Alım Tahmin Sistemi

