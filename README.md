# Restoran Yemek Fiyatı ve Garson Performansı Ölçme Sistemi

**Akıllı restoran yönetim sistemi** - QR kod tabanlı masa takibi, YOLOv8 yemek tespiti ve gerçek zamanlı garson performans analizi.

## Proje Özeti

Bu proje bir restoranda masa üzerinde bulunan kameralar aracılığıyla yemek tanıma, fiyatlandırma ve garson performans ölçümü yapan tam otomatik akıllı sistemdir.

### Ana Özellikler

- **QR Kod Tabanlı Masa Takibi** - Müşteri gelişi/gidişi otomatik tespiti
- **YOLOv8 Yemek Tespiti** - Google Colab'da eğitilmiş özel model
- **Garson Performans Ölçümü** - Adil puanlama sistemi
- **Otomatik Hesap Yönetimi** - Anlık fiyatlandırma ve hesap sıfırlama
- **Türkçe Lokalizasyon** - Tam Türkçe arayüz ve raporlama

## Hızlı Başlangıç

### 1. Gereksinimler

```bash
# Virtual environment oluştur (önerilen)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Sistem Çalıştırma

```bash
python main.py
```

### 3. Demo Video Seçimi

Sistem başladığında size demo video seçenekleri sunulacak:

- **Demo Video 1**: Birinci açı
- **Demo Video 2**: İkinci açı
- **Kendi Videonuz**: Özel video dosyası

### 4. Kontroller

- **[ESC]** - Sistemden çıkış
- **[SPACE]** - Video duraklat/devam et
- **[R]** - Video başa dön
- **[C]** - Hesap sıfırla
- **[Q]** - Hızlı çıkış

## Sistem Bileşenleri

### Sistem Mimarisi

1. **Ana Koordinatör** (`main.py`)

   - Video işleme ve sistem entegrasyonu
   - QR kod tespiti (masa + garson)
   - Tüm modüllerin koordinasyonu

2. **YOLOv8 Yemek Tespiti** (`yolo_food_detector.py`)

   - Google Colab'da eğitilmiş özel model
   - Yenek ve tabak tespiti
   - Duplikasyon önleme algoritması

3. **Masa Yönetimi** (`table_manager.py`)

   - 4 masa durumu takibi (EMPTY, WAITING, SERVED)
   - Garson performans puanlaması
   - Zamanlayıcı sistemi

4. **Garson Takibi** (`waiter_detector.py`)
   - QR kod ile garson konumu belirleme
   - Masa-garson eşleştirmesi
   - Yanıt süresi ölçümü

### Core Files

- `main.py` - Ana sistem koordinatörü
- `yolo_food_detector.py` - YOLOv8 yemek tespit sistemi
- `table_manager.py` - Masa ve garson yönetimi
- `waiter_detector.py` - Garson konumu takibi

### Model & Data

- `models/food_detection.pt` - Eğitilmiş YOLOv8 modeli (6.2MB)
- `demo/demo_video.mov` - Demo video (Birinci açı)
- `demo/demo_video2.mp4` - Demo video (İkinci açı)

### Kütüphaneler

- `requirements.txt` - Proje bağımlılıkları

## QR Kod Sistemi

### Masa Kodları

- `m001` → MASA_1 (GARSON_1 sorumlusu)
- `m002` → MASA_2 (GARSON_1 sorumlusu)
- `m003` → MASA_3 (GARSON_2 sorumlusu)
- `m004` → MASA_4 (GARSON_2 sorumlusu)

### Garson Kodları

- `g001` → GARSON_1
- `g002` → GARSON_2

### Çalışma Mantığı

1. **Masa QR kodu görünür** → Masa boş
2. **Müşteri gelir, QR kodu ters çevirir** → Masa dolu, zamanlayıcı başlar
3. **Garson QR kodu okunur** → Servis süresi kaydedilir
4. **Yemekler getirilir** → YOLO model ile tespit ve fiyatlandırma
5. **Müşteri kalkar, QR kodu geri koyar** → Hesap sıfırlanır

## Yemek Tespiti (YOLOv8)

### Model Detayları

- **Eğitim Platform**: Google Colab (Tesla T4 GPU)
- **Model Tipi**: YOLOv8n (nano - hızlı tespit)

## Garson Performans Sistemi

### Puanlama

- **Base Score**: 100 puan
- **Penalty**: 60+ saniye bekleme → -18.2 puan
- **Fair Policy**: 60 saniye altı beklemede puan düşmez

### Metrikler

- Ortalama yanıt süresi
- Toplam servis sayısı
- Uyarı sayısı
- Performans skoru (0-100)

### Hesaplama Formülü

```python
waiting_time_penalty = max(0, (waiting_time - 60.0) / 5.0)
score = max(0, 100 - (warnings * waiting_time_penalty))
```

## Sistem Çıktıları

### Video Görselleştirme

- QR kod bounding box'ları
- Yemek tespit çerçeveleri
- Masa durumu gösterimi
- Anlık hesap bilgisi
- Garson performans özeti

## Proje Yapısı

```

goru-final/
├── main.py # Ana sistem
├── yolo_food_detector.py # YOLO yemek tespiti
├── table_manager.py # Masa yönetimi
├── waiter_detector.py # Garson takibi
├── models/
│ └── food_detection.pt # Eğitilmiş model (6.2MB)
├── demo/
│ ├── demo_video.mov # Demo video 1
│ └── demo_video2.mp4 # Demo video 2
├── requirements.txt # Proje bağımlılıkları
└── README.md # Bu dosya - Tam dokümantasyon

```

## Gelişmiş Özellikler

### Duplikasyon Önleme Algoritması

**Çok Katmanlı Filtre:**

```python
1. Distance Threshold: 120 piksel (aynı yemekler birbirine yakın olamaz)
2. Stability Frames: 3 frame (tutarlı tespit gerekli)
3. Category Matching: Aynı kategori kontrol
4. Time-based Cleanup: 5 saniye eski tespitleri temizle
```

**Sonuç:** %60 duplikasyon azalması

### Duplikasyon Önleme

- Distance threshold: 120 piksel
- Stability frames: 3 frame
- Time-based cleanup: 5 saniye

### Video İşleme

- Otomatik boyutlandırma (1200x675)
- FPS kontrolü (30 FPS)
- Pencere yönetimi
- Klavye kontrolları

### Error Handling

- Dosya varlığı kontrolü
- Video format desteği
- Graceful shutdown
- Exception handling

## Teknik Gereksinimler

### Donanım

- **Kamera:** HD webcam veya video dosyası
- **İşlemci:** Orta seviye CPU (Intel i5/AMD Ryzen 5+)
- **RAM:** 8GB+ (video processing için)
- **GPU:** İsteğe bağlı (YOLO hızlandırma)

### Yazılım Stack

```
Python 3.8+
├── OpenCV 4.5+ (video processing)
├── ultralytics (YOLOv8)
├── pyzbar (QR decoding)
└── numpy (numerical operations)
```

## Gelecek Geliştirmeler

### Model İyileştirmeleri

- **Daha fazla yemek kategorisi**
- **Portion detection:** Yemek boyutu tanıma
- **Quality assessment:** Yemek kalitesi değerlendirme

### Sistem Genişletmeleri

- **Web dashboard** (Gerçek zamanlı)
- **Mobile app** (Garson uygulaması)
- **IoT integration** (Fiziksel sensörler)

### Analitik Özellikler

- **Revenue tracking:** Gelir analizi
- **Customer behavior:** Müşteri davranış analizi
- **Inventory management:** Stok takibi

---

**Not**: Bu proje Bilgisayarlı Görüye Giriş dersi Final projesi olarak geliştirilmiştir.
