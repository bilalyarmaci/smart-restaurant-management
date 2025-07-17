"""
YOLOv8 tabanlı Yemek Tespit Sistemi
Roboflow'dan eğitilen model ile entegrasyon
"""

import cv2
import numpy as np
from datetime import datetime
import os
from ultralytics import YOLO

class YOLOFoodDetector:
    def __init__(self, model_path='models/food_detection.pt'):
        """
        YOLOv11/YOLOv8 tabanlı yemek detector'ı başlat
        """
        # Yemek kategorileri ve fiyatları - Custom Model (plate + pogaca)
        self.food_categories = {
            0: {  # YOLO class index 0 - plate (tabak)
                'name': 'Tabak',
                'price': 0.0,  # Tabak için ücret yok
                'color': (200, 200, 200),  # Gri
                'category': 'plate'
            },
            1: {  # YOLO class index 1 - pogaca
                'name': 'Pogaca',
                'price': 15.0,
                'color': (0, 127, 255),  # Açık turuncu
                'category': 'pogaca'
            }
        }
        
        # Model yolu
        self.model_path = model_path
        self.model = None
        
        # Tespit edilen yemekler (masa bazlı)
        self.detected_foods = {}
        
        # Geçici tespit buffer'ı (stability için)
        self.temp_detections = {}
        
        # Tespit parametreleri
        self.confidence_threshold = 0.5  # YOLOv8 confidence threshold
        self.duplicate_distance_threshold = 120  # Daha büyük mesafe (daha iyi takip)
        self.stability_frames = 3  # Daha fazla frame bekle (daha güvenilir)
        
        # Model yükle
        self.load_model()
        
        print("🤖 YOLOv8 Yemek Tespit Sistemi baslatildi")
      
    
    def load_model(self):
        """
        YOLOv8 modelini yükle
        """
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"✅ Model yüklendi: {self.model_path}")
                
            except Exception as e:
                print(f"❌ Model yüklenemedi: {e}")
                print("🔄 Pre-trained YOLOv11n modeli kullanılacak (genel amaçlı)")
                self.model = YOLO('yolo11n.pt')  # YOLOv11 nano model
        else:
            print(f"⚠️ Model dosyası bulunamadı: {self.model_path}")
            print("📁 Beklenen konum: models/food_detection.pt")
            print("🔄 Pre-trained YOLOv11n modeli kullanılacak (genel amaçlı)")
            
            # Pre-trained model ile devam et
            self.model = YOLO('yolo11n.pt')  # YOLOv11 nano model
            
            # Genel model için food kategorilerini güncelle
            self._setup_pretrained_categories()
    
    def _setup_pretrained_categories(self):
        """
        Pre-trained model için genel kategorileri ayarla
        """
        # COCO dataset'inden yemek ile ilgili kategoriler
        coco_food_mapping = {
            47: {'name': 'Kupa/Kase', 'price': 0.0, 'color': (255, 255, 0), 'category': 'cup'},
            48: {'name': 'Çatal', 'price': 0.0, 'color': (0, 255, 255), 'category': 'fork'},
            49: {'name': 'Bıçak', 'price': 0.0, 'color': (255, 0, 255), 'category': 'knife'},
            50: {'name': 'Kaşık', 'price': 0.0, 'color': (0, 255, 0), 'category': 'spoon'},
            51: {'name': 'Kase', 'price': 0.0, 'color': (255, 0, 0), 'category': 'bowl'},
            52: {'name': 'Muz', 'price': 5.0, 'color': (0, 255, 255), 'category': 'banana'},
            53: {'name': 'Elma', 'price': 3.0, 'color': (0, 0, 255), 'category': 'apple'},
            54: {'name': 'Sandviç', 'price': 25.0, 'color': (255, 255, 0), 'category': 'sandwich'},
            55: {'name': 'Portakal', 'price': 4.0, 'color': (0, 165, 255), 'category': 'orange'},
            56: {'name': 'Brokoli', 'price': 8.0, 'color': (0, 255, 0), 'category': 'broccoli'},
            57: {'name': 'Havuç', 'price': 6.0, 'color': (0, 127, 255), 'category': 'carrot'},
            58: {'name': 'Pizza', 'price': 45.0, 'color': (255, 0, 0), 'category': 'pizza'},
            59: {'name': 'Donut', 'price': 15.0, 'color': (255, 0, 255), 'category': 'donut'},
            60: {'name': 'Kek', 'price': 35.0, 'color': (255, 255, 0), 'category': 'cake'}
        }
        
        print("🔄 Pre-trained model food kategorileri:")
        for class_id, info in coco_food_mapping.items():
            print(f"   • {info['name']}: {info['price']} TL")
        
        self.food_categories = coco_food_mapping
    
    def detect_food_on_frame(self, frame, table_areas=None):
        """
        YOLOv8 ile frame'de yemek tespiti yap
        """
        detected_items = []
        
        if self.model is None:
            return detected_items
        
        try:
            # YOLOv8 ile tespit yap
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Sonuçları işle
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Box bilgilerini al
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Yemek kategorisi kontrolü
                        if class_id in self.food_categories:
                            category_info = self.food_categories[class_id]
                            
                            # Bounding box formatını ayarla
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            
                            detected_items.append({
                                'category': category_info['category'],
                                'class_id': class_id,
                                'bbox': (x, y, w, h),
                                'center': (x + w//2, y + h//2),
                                'area': w * h,
                                'confidence': float(confidence),
                                'name': category_info['name'],
                                'price': category_info['price'],
                                'color': category_info['color'],
                                'timestamp': datetime.now()
                            })
            
        except Exception as e:
            print(f"❌ YOLO tespit hatası: {e}")
        
        return detected_items
    
    def update_table_food_status(self, table_id, detected_foods):
        """
        Masa bazlı yemek durumunu güncelle - YOLOv8 için optimize edilmiş
        """
        if table_id not in self.detected_foods:
            self.detected_foods[table_id] = {
                'items': [],
                'total_price': 0.0,
                'last_update': datetime.now()
            }
        
        if table_id not in self.temp_detections:
            self.temp_detections[table_id] = []
        
        current_items = self.detected_foods[table_id]['items']
        temp_items = self.temp_detections[table_id]
        
        # Mevcut temp detections'ları güncelle
        for food in detected_foods:
            if food['price'] == 0.0:  # Tabak/çatal gibi fiyatsız itemları atla
                continue
                
            food_center = food['center']
            food_category = food['category']
            food_confidence = food['confidence']
            
            # 1. Mevcut confirmed itemlarda çok yakın benzer yemek var mı?
            is_already_confirmed = False
            for existing_item in current_items:
                existing_center = existing_item['center']
                distance = np.sqrt((food_center[0] - existing_center[0])**2 + 
                                 (food_center[1] - existing_center[1])**2)
                
                # Hem aynı kategori hem de yakın mesafe
                if distance < self.duplicate_distance_threshold and food_category == existing_item['category']:
                    is_already_confirmed = True
                    # Mevcut item'ın pozisyonunu güncelle (hareket takibi)
                    existing_item['center'] = food_center
                    existing_item['last_seen'] = datetime.now()
                    break
            
            if not is_already_confirmed:
                # 2. Temp detections'da benzer yemek var mı?
                found_in_temp = False
                for temp_item in temp_items:
                    temp_center = temp_item['center']
                    distance = np.sqrt((food_center[0] - temp_center[0])**2 + 
                                     (food_center[1] - temp_center[1])**2)
                    
                    # Hem aynı kategori hem de yakın mesafe
                    if distance < self.duplicate_distance_threshold and food_category == temp_item['category']:
                        # Mevcut temp item'ı güncelle
                        temp_item['count'] += 1
                        temp_item['last_seen'] = datetime.now()
                        temp_item['center'] = food_center  # Pozisyon güncelle
                        temp_item['confidence'] = max(temp_item['confidence'], food_confidence)
                        found_in_temp = True
                        break
                
                if not found_in_temp:
                    # 3. Yeni temp detection ekle
                    food['count'] = 1
                    food['last_seen'] = datetime.now()
                    temp_items.append(food)
        
        # Stability kontrolü
        items_to_confirm = []
        remaining_temp_items = []
        
        for temp_item in temp_items:
            if temp_item['count'] >= self.stability_frames:
                items_to_confirm.append(temp_item)
            else:
                remaining_temp_items.append(temp_item)
        
        # Confirmed items'ları ana listeye ekle
        for item in items_to_confirm:
            current_items.append(item)
            print(f"🍽️ {table_id}: {item['name']} onaylandi! (+{item['price']:.0f} TL) [Confidence: {item['confidence']:.2f}]")
        
        # Temp detections listesini güncelle
        self.temp_detections[table_id] = remaining_temp_items
        
        # Eski temp detections'ları temizle
        current_time = datetime.now()
        self.temp_detections[table_id] = [
            item for item in self.temp_detections[table_id]
            if (current_time - item['last_seen']).total_seconds() < 5
        ]
        
        # Çok eski confirmed items'ları da temizle (10 saniyeden eski ve artık görünmeyen)
        # Bu sadece eğer masa üzerinde hiç yemek tespit edilmiyorsa yapılır
        if len(detected_foods) == 0:  # Hiç yemek tespit edilmiyor
            current_items[:] = [
                item for item in current_items
                if hasattr(item, 'last_seen') and (current_time - item['last_seen']).total_seconds() < 10
            ]
        
        # Toplam fiyatı hesapla
        total_price = sum(item['price'] for item in current_items)
        self.detected_foods[table_id]['total_price'] = total_price
        self.detected_foods[table_id]['last_update'] = datetime.now()
        
        return len(current_items), total_price
    
    def detect_plates_and_bowls(self, frame):
        """
        Tabak ve kase tespiti - YOLO model üzerinden
        Model zaten plate sınıfını tespit ediyor
        """
        plates = []
        
        try:
            # YOLO prediction
            results = self.model(frame, conf=0.3, verbose=False)  # Düşük confidence tabaklar için
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        # Sadece plate (class 0) sınıfını al
                        if class_id == 0 and confidence > 0.3:  # plate
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            radius = int(max(x2 - x1, y2 - y1) / 2)
                            
                            plates.append({
                                'center': (center_x, center_y),
                                'radius': radius,
                                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'area': (x2 - x1) * (y2 - y1),
                                'type': 'plate',
                                'confidence': confidence,
                                'timestamp': datetime.now()
                            })
                            
        except Exception as e:
            print(f"❌ Plate detection hatası: {e}")
        
        return plates

    def draw_food_detections(self, frame, detected_foods, plates=None):
        """
        Tespit edilen yemekleri frame üzerine çiz
        """
        # Tabak/kaseleri çiz
        if plates:
            for plate in plates:
                center = plate['center']
                radius = plate['radius']
                confidence = plate.get('confidence', 0.0)
                
                # Tabak çemberi
                color = (200, 200, 200)  # Gri
                cv2.circle(frame, center, radius, color, 2)
                
                # Merkez nokta
                cv2.circle(frame, center, 3, color, -1)
                
                # Label
                label = f"PLATE ({confidence:.2f})"
                cv2.putText(frame, label, (center[0] - 30, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Yemekleri çiz
        for food in detected_foods:
            x, y, w, h = food['bbox']
            name = food['name']
            price = food['price']
            confidence = food['confidence']
            color = food['color']
            
            # Bounding box çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Label için arka plan
            if price > 0:  # Sadece ücretli itemlar için fiyat göster
                label = f"{name}: {price} TL ({confidence:.2f})"
            else:
                label = f"{name} ({confidence:.2f})"
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Label yazısı
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Merkez çizgisi
            center = food['center']
            cv2.circle(frame, center, 5, color, -1)
        
        return frame
    
    def draw_table_bill(self, frame, table_id, position=(10, 200)):
        """
        Masa hesabını frame üzerine çiz
        """
        if table_id not in self.detected_foods:
            return frame
        
        table_data = self.detected_foods[table_id]
        items = table_data['items']
        total_price = table_data['total_price']
        
        x, y = position
        
        # Başlık
        cv2.putText(frame, f"{table_id} HESABI:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        
        # Yemek listesi
        item_counts = {}
        for item in items:
            name = item['name']
            price = item['price']
            
            if price > 0:  # Sadece ücretli itemları say
                if name in item_counts:
                    item_counts[name]['count'] += 1
                    item_counts[name]['total'] += price
                else:
                    item_counts[name] = {'count': 1, 'total': price, 'unit_price': price}
        
        # Her yemek türünü listele
        for name, data in item_counts.items():
            count = data['count']
            total = data['total']
            
            item_text = f"{count}x {name}: {total:.0f} TL"
            cv2.putText(frame, item_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 20
        
        # Toplam
        y += 10
        cv2.putText(frame, f"TOPLAM: {total_price:.0f} TL", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def clear_table_bill(self, table_id):
        """
        Masa hesabını sıfırla (QR kod tekrar okunduğunda)
        """
        old_total = 0.0
        
        if table_id in self.detected_foods:
            old_total = self.detected_foods[table_id]['total_price']
            self.detected_foods[table_id] = {
                'items': [],
                'total_price': 0.0,
                'last_update': datetime.now()
            }
        
        # Temp detections'ı da temizle
        if table_id in self.temp_detections:
            self.temp_detections[table_id] = []
        
        if old_total > 0:
            print(f"🧾 {table_id}: Hesap sifirlandi (Onceki total: {old_total:.0f} TL)")
        
        return old_total
    
    def get_table_summary(self, table_id):
        """
        Masa özet bilgilerini al
        """
        if table_id not in self.detected_foods:
            return None
        
        table_data = self.detected_foods[table_id]
        
        # Yemek sayıları (sadece ücretli itemlar)
        item_counts = {}
        for item in table_data['items']:
            if item['price'] > 0:  # Sadece ücretli itemları say
                name = item['name']
                if name in item_counts:
                    item_counts[name] += 1
                else:
                    item_counts[name] = 1
        
        return {
            'table_id': table_id,
            'items': item_counts,
            'total_items': len([item for item in table_data['items'] if item['price'] > 0]),
            'total_price': table_data['total_price'],
            'last_update': table_data['last_update']
        }

    def get_all_tables_summary(self):
        """
        Tüm masaların özet bilgilerini al
        """
        summaries = []
        for table_id in self.detected_foods:
            summary = self.get_table_summary(table_id)
            if summary:
                summaries.append(summary)
        return summaries

# Test fonksiyonu
def test_yolo_detector():
    """
    YOLOv8 detector'ı test et
    """
    detector = YOLOFoodDetector()
    
    print("\n🧪 YOLOv8 Food Detector Test")
    print(f"📊 Model: {detector.model_path}")
    print(f"🎯 Confidence threshold: {detector.confidence_threshold}")
    
    # Test görüntüsü ile test
    if os.path.exists('test_frame.jpg'):
        print("\n📸 Test görüntüsü ile test ediliyor...")
        frame = cv2.imread('test_frame.jpg')
        foods = detector.detect_food_on_frame(frame)
        print(f"🍽️ Tespit edilen yemek sayısı: {len(foods)}")
        
        for food in foods:
            print(f"   • {food['name']}: {food['confidence']:.2f}")
    
    return detector

if __name__ == "__main__":
    test_yolo_detector()
