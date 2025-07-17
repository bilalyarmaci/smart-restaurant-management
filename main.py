"""
Yemek Fiyatı ve Garson Performansı Ölçme Sistemi
Adım 1-2: Video Analizi, QR Kod Tespiti ve Zamanlayıcı Sistemi
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import time
from datetime import datetime
import json
from table_manager import TableManager, TableStatus
from waiter_detector import EnhancedWaiterDetector
from yolo_food_detector import YOLOFoodDetector

class QRCodeDetector:
    def __init__(self):
        # TableManager entegrasyonu
        self.table_manager = TableManager()
        self.waiter_detector = EnhancedWaiterDetector()
        self.food_detector = YOLOFoodDetector()
        
        # QR kod tipleri - Demo video uyumlu
        self.table_qr_codes = [
            "MASA_1", "MASA_2", "MASA_3", "MASA_4",  # Standart format
            "m001", "m002", "m003", "m004"  # Demo video formatı (masa kodları)
        ]
        self.waiter_qr_codes = [
            "GARSON_1", "GARSON_2",  # Standart format
            "w001", "w002", "g001", "g002"  # Demo video formatı (garson kodları)
        ]
        
        # QR kod çeviri haritası
        self.qr_translation = {
            "m001": "MASA_1",
            "m002": "MASA_2", 
            "m003": "MASA_3",
            "m004": "MASA_4",
            "w001": "GARSON_1",
            "w002": "GARSON_2",
            "g001": "GARSON_1",  # garson 1
            "g002": "GARSON_2"   # garson 2
        }
        
        # QR detection counters
        self.table_detection_counts = {
            "MASA_1": 0,
            "MASA_2": 0,
            "MASA_3": 0,
            "MASA_4": 0
        }
        
        # Previous states for change detection
        self.previous_table_states = {}
        self.previous_waiter_states = {}
        
        # Eski sistem uyumluluğu için
        self.table_states = self.table_manager.tables
        
    def detect_qr_codes(self, frame):
        """
        Frame'de QR kodları tespit et - Gelişmiş versiyon
        Farklı açılardan ve rotasyonlarda QR kodları okuyabilir
        """
        qr_codes = []
        
        # Orijinal frame'i dene
        decoded_objects = pyzbar.decode(frame)
        qr_codes.extend(self._process_decoded_objects(decoded_objects))
        
        # Gri tonlamaya çevir (daha iyi tespit için)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)
        qr_codes.extend(self._process_decoded_objects(decoded_objects))
        
        # Kontrast artırma
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        decoded_objects = pyzbar.decode(enhanced)
        qr_codes.extend(self._process_decoded_objects(decoded_objects))
        
        # Gaussian blur (gürültü azaltma)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        decoded_objects = pyzbar.decode(blurred)
        qr_codes.extend(self._process_decoded_objects(decoded_objects))
        
        # Farklı rotasyonları dene (garson QR kodları için)
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(gray, angle)
            decoded_objects = pyzbar.decode(rotated)
            rotated_qr_codes = self._process_decoded_objects(decoded_objects, rotation=angle)
            qr_codes.extend(rotated_qr_codes)
        
        # Duplikasyonları temizle (aynı QR kod farklı yöntemlerle tespit edilebilir)
        unique_qr_codes = self._remove_duplicate_qr_codes(qr_codes)
        
        return unique_qr_codes
    
    def _process_decoded_objects(self, decoded_objects, rotation=0):
        """
        Decode edilmiş QR objelerini işle
        """
        qr_codes = []
        
        for obj in decoded_objects:
            try:
                # QR kod verisi
                qr_data = obj.data.decode('utf-8')
                
                # QR kod pozisyonu
                points = obj.polygon
                if len(points) == 4:
                    # Bounding box koordinatları
                    x = min([p.x for p in points])
                    y = min([p.y for p in points])
                    w = max([p.x for p in points]) - x
                    h = max([p.y for p in points]) - y
                    
                    qr_codes.append({
                        'data': qr_data,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'rotation': rotation,
                        'confidence': 1.0,  # pyzbar her zaman 1.0 döner
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                print(f"⚠️ QR kod işleme hatası: {e}")
                continue
        
        return qr_codes
    
    def _rotate_image(self, image, angle):
        """
        Görüntüyü belirtilen açıda döndür
        """
        if angle == 0:
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotasyon matrisini oluştur
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Yeni boyutları hesapla
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Çeviri değerlerini ayarla
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Döndürülmüş görüntüyü döndür
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated
    
    def _remove_duplicate_qr_codes(self, qr_codes):
        """
        Aynı QR kodun farklı yöntemlerle tespit edildiği duplikasyonları temizle
        """
        unique_codes = []
        seen_data = set()
        
        # En yüksek confidence'a sahip olanları tercih et
        sorted_codes = sorted(qr_codes, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for qr in sorted_codes:
            if qr['data'] not in seen_data:
                unique_codes.append(qr)
                seen_data.add(qr['data'])
        
        return unique_codes
    
    def draw_qr_codes(self, frame, qr_codes):
        """
        QR kodları frame üzerine çiz - Demo video uyumlu
        """
        for qr in qr_codes:
            x, y, w, h = qr['bbox']
            qr_data = qr['data']
            translated = self._translate_qr_code(qr_data)
            
            # QR kod tipine göre renk ve stil seç
            if self._is_table_qr(qr_data):
                color = (0, 255, 0)  # Yeşil - Masa QR
                label = f"MASA: {translated}"
                thickness = 3
            elif self._is_waiter_qr(qr_data):
                color = (0, 0, 255)  # Kırmızı - Garson QR  
                label = f"GARSON: {translated}"
                thickness = 4
                
                # Garson QR kod için özel işaretleme
                cv2.circle(frame, qr['center'], 10, (0, 255, 255), -1)  # Merkez nokta
            else:
                color = (255, 255, 255)  # Beyaz - Bilinmeyen QR
                label = f"UNKNOWN: {qr_data[:10]}"
                thickness = 2
            
            # Bounding box çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Rotasyon bilgisi varsa göster
            if qr.get('rotation', 0) != 0:
                label += f" (Rot:{qr['rotation']}°)"
            
            # Label için arka plan
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Label yazısı
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Merkez çizgisi
            cv2.line(frame, (qr['center'][0] - 10, qr['center'][1]), 
                    (qr['center'][0] + 10, qr['center'][1]), color, 2)
            cv2.line(frame, (qr['center'][0], qr['center'][1] - 10), 
                    (qr['center'][0], qr['center'][1] + 10), color, 2)
        
        return frame
    
    def update_table_states(self, qr_codes):
        """
        Tespit edilen QR kodlara göre masa durumlarını güncelle - Demo video uyumlu
        """
        # Reset detection counts
        for table in self.table_detection_counts:
            self.table_detection_counts[table] = 0
        
        # QR kod verilerini çıkar ve çevir
        table_qr_data = []
        waiter_detections = []
        
        for qr in qr_codes:
            qr_data = qr['data']
            translated = self._translate_qr_code(qr_data)
            
            if self._is_table_qr(qr_data):
                table_qr_data.append(translated)  # Çevrilmiş versiyonu kullan
                if translated in self.table_detection_counts:
                    self.table_detection_counts[translated] += 1
                
                # Only print if this is a state change (check after table manager update)
                table_key = f"qr_detected_{translated}"
                if table_key not in self.previous_table_states:
                    print(f"🔍 Masa QR tespit edildi: {qr_data} → {translated}")
                    self.previous_table_states[table_key] = True
                    
                    # Masa QR kodu tekrar okunursa hesabı sıfırla
                    if translated in self.food_detector.detected_foods:
                        self.food_detector.clear_table_bill(translated)
                    
            elif self._is_waiter_qr(qr_data):
                waiter_detections.append({
                    'waiter_id': translated,  # Çevrilmiş versiyonu kullan
                    'original_id': qr_data,
                    'position': qr['center'],
                    'timestamp': qr['timestamp']
                })
                
                # Only print waiter detection if it's a new detection or position change
                waiter_key = f"{translated}_{qr['center']}"
                if waiter_key not in self.previous_waiter_states:
                    print(f"👨‍💼 Garson QR tespit edildi: {qr_data} → {translated}")
                    self.previous_waiter_states[waiter_key] = True
        
        # TableManager ile masa durumlarını güncelle
        previous_table_states_snapshot = {}
        for table_name, table_data in self.table_manager.tables.items():
            previous_table_states_snapshot[table_name] = table_data["status"]
        
        self.table_manager.update_table_qr_status(table_qr_data)
        
        # Masa durumu değişimlerini kontrol et - yeni müşteri geldiğinde hesap sıfırla
        for table_name, table_data in self.table_manager.tables.items():
            current_status = table_data["status"]
            previous_status = previous_table_states_snapshot.get(table_name)
            
            # Müşteri geldi durumu: EMPTY -> WAITING
            if previous_status == TableStatus.EMPTY and current_status == TableStatus.WAITING:
                table_id = table_name.replace("table_", "MASA_")  # table_1 -> MASA_1
                if table_id in self.food_detector.detected_foods:
                    old_total = self.food_detector.clear_table_bill(table_id)
                    if old_total > 0:
                        print(f"🧾 {table_id}: Yeni müşteri - hesap sıfırlandı (Önceki: {old_total:.0f} TL)")
        
        # Garson tespitlerini işle - Enhanced Waiter Detector kullan
        for waiter in waiter_detections:
            # Enhanced waiter detector ile işle
            detected_waiter = self.waiter_detector.process_waiter_qr(
                waiter['original_id'], 
                waiter['position'], 
                waiter['timestamp']
            )
            
            # Eski sistem uyumluluğu için TableManager'a da bildir
            table_served, response_time = self.table_manager.waiter_detected(
                waiter['waiter_id'], 
                waiter['position']
            )
            if table_served:
                print(f"✅ {waiter['original_id']} ({waiter['waiter_id']}) → {table_served.upper()}: {response_time:.1f}s")
        
        # Uyarı kontrolü (60 saniye) - sadece yeni uyarılar için
        warnings = self.table_manager.check_warnings(60)
        # Uyarılar zaten TableManager tarafından yazdırılıyor
    
    def detect_waiters(self, qr_codes):
        """
        Garson QR kodlarını tespit et
        """
        waiters_detected = []
        
        for qr in qr_codes:
            if qr['data'] in self.waiter_qr_codes:
                waiters_detected.append({
                    'waiter_id': qr['data'],
                    'position': qr['center'],
                    'timestamp': qr['timestamp']
                })
        
        return waiters_detected
    
    def draw_table_status(self, frame):
        """
        Masa durumlarını frame üzerine yazdır - Demo video uyumlu
        """
        y_offset = 30
        
        # Masa durumları - tüm masaları göster
        cv2.putText(frame, "MASA DURUMU:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # MASA_1 için gerçek durum bilgisi al
        status_list = self.table_manager.get_table_status_display()
        table1_status = None
        for status in status_list:
            if status['table'] == 'MASA_1':
                table1_status = status
                break
        
        # MASA_1 durumunu göster
        if table1_status:
            table_name = table1_status['table']
            table_status = table1_status['status']
            waiting_time = table1_status['waiting_time']
            assigned_waiter = table1_status['assigned_waiter']
            
            # Durum metnini oluştur
            if table_status == "empty":
                status_text = f"{table_name}: BOS"
                color = (0, 255, 0)  # Yeşil
            elif table_status == "waiting":
                status_text = f"{table_name}: BEKLIYOR ({waiting_time:.0f}s)"
                color = (0, 255, 255)  # Sarı
                if waiting_time > 60:
                    color = (0, 0, 255)  # Kırmızı (60s+)
            elif table_status == "served":
                status_text = f"{table_name}: SERVIS"
                color = (255, 0, 255)  # Magenta
            else:  # occupied
                status_text = f"{table_name}: DOLU"
                color = (0, 0, 255)  # Kırmızı
            
            # Garson bilgisi ekle
            if assigned_waiter:
                status_text += f" ({assigned_waiter})"
        else:
            # MASA_1 için varsayılan
            count = self.table_detection_counts.get("MASA_1", 0)
            if count > 0:
                status_text = f"MASA_1: Detected ({count})"
                color = (0, 255, 0)  # Yeşil
            else:
                status_text = f"MASA_1: Tespit Edilmedi"
                color = (0, 0, 255)  # Kırmızı
        
        cv2.putText(frame, status_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
        
        # MASA_2, MASA_3, MASA_4 için Unknown durumu
        for table_id in ["MASA_2", "MASA_3", "MASA_4"]:
            status_text = f"{table_id}: Bilinmiyor"
            color = (128, 128, 128)  # Gri - Unknown
            
            cv2.putText(frame, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Garson performans özeti - sadece aktif garsonlar
        y_offset += 10
        cv2.putText(frame, "GARSON PERFORMANSI:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        performance = self.table_manager.get_performance_summary()
        for waiter_id, perf in performance.items():
            # Sadece servis yapmış garsonları göster
            if perf['total_services'] > 0:
                perf_text = f"{waiter_id}: Score:{perf['performance_score']} Avg:{perf['avg_response']}s"
                
                # Performans rengini belirle
                score = perf['performance_score']
                if score >= 80:
                    perf_color = (0, 255, 0)  # Yeşil - İyi
                elif score >= 60:
                    perf_color = (0, 255, 255)  # Sarı - Orta
                else:
                    perf_color = (0, 0, 255)  # Kırmızı - Kötü
                
                cv2.putText(frame, perf_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
                y_offset += 20
        
        return frame
    
    def process_video(self, video_path):
        """
        Video dosyasını işle - Gelişmiş sürüm
        """
        # Video dosyasının varlığını kontrol et
        import os
        if not os.path.exists(video_path):
            print(f"❌ Video dosyası bulunamadı: {video_path}")
            print(f"💡 Lütfen video dosyasını '{video_path}' konumuna yerleştirin")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Video dosyası açılamadı: {video_path}")
            print("💡 Video formatının desteklendiğinden emin olun (mp4, avi, mov)")
            return None
        
        # Video bilgilerini al
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count_total / fps if fps > 0 else 0
        
        # Ekran boyutuna uygun olarak yeniden boyutlandırma
        max_width = 1200
        max_height = 800
        scale_factor = min(max_width/width, max_height/height, 1.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"🎥 Video Bilgileri:")
        print(f"   📁 Dosya: {video_path}")
        print(f"   📐 Orijinal Boyut: {width}x{height}")
        print(f"   📐 Görüntüleme Boyutu: {new_width}x{new_height}")
        print(f"   ⏱️ FPS: {fps}")
        print(f"   🎬 Toplam Frame: {frame_count_total}")
        print(f"   ⏰ Süre: {duration:.1f} saniye")
        print(f"\n🔍 QR kod tespiti başlatıldı...")
        print(f"🍽️ Yemek tespit sistemi aktif...")
        print(f"📋 Masa durumları takip ediliyor...")
        print(f"\n[ESC] ile çıkış, [SPACE] ile duraklat/devam et, [R] ile başa dön, [C] ile hesap sıfırla\n")
        
        frame_count = 0
        paused = False
        qr_codes = []  # QR kodları için başlangıç değeri
        
        # OpenCV pencere ayarları
        window_name = 'Restaurant QR Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, new_width, new_height)
        
        # Ekran ortasına konumlandır
        screen_width = 1920  # Varsayılan ekran genişliği
        screen_height = 1080  # Varsayılan ekran yüksekliği
        x_pos = (screen_width - new_width) // 2
        y_pos = (screen_height - new_height) // 2
        cv2.moveWindow(window_name, x_pos, y_pos)
        
        # Pencere her zaman görünür olsun
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # FPS kontrolü için zamanlayıcı
        import time
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Minimum 30 FPS
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            
            if not paused and (current_time - last_frame_time) >= frame_delay:
                ret, frame = cap.read()
                
                if not ret:
                    print("📹 Video sonu ulaşıldı - başa dönmek için [R] tuşuna basın")
                    paused = True
                    continue
                
                frame_count += 1
                last_frame_time = current_time
                
                # Frame'i yeniden boyutlandır
                if scale_factor < 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Her 2 frame'de bir QR kod tespiti yap (daha sık kontrol)
                if frame_count % 2 == 0:
                    # QR kodları tespit et
                    qr_codes = self.detect_qr_codes(frame)
                    
                    # Masa durumlarını güncelle
                    self.update_table_states(qr_codes)
                    
                    # Yemek tespiti yap (her 5 frame'de bir)
                    if frame_count % 10 == 0:
                        detected_foods = self.food_detector.detect_food_on_frame(frame)
                        plates = self.food_detector.detect_plates_and_bowls(frame)
                        
                        # MASA_1 için yemek durumunu güncelle
                        if detected_foods:
                            food_count, total_price = self.food_detector.update_table_food_status('MASA_1', detected_foods)
                    
                    # Garsonları tespit et
                    waiters = self.detect_waiters(qr_codes)
                    if waiters:
                        # Only print if it's a new waiter detection at a different position
                        for waiter in waiters:
                            waiter_pos_key = f"{waiter['waiter_id']}_{waiter['position'][0]}_{waiter['position'][1]}"
                            if waiter_pos_key not in self.previous_waiter_states:
                                print(f"👨‍💼 {waiter['waiter_id']} tespit edildi! Pozisyon: {waiter['position']}")
                                self.previous_waiter_states[waiter_pos_key] = True
                                
                                # Clear old position states for this waiter (keep only recent ones)
                                keys_to_remove = []
                                for key in self.previous_waiter_states:
                                    if key.startswith(waiter['waiter_id']) and key != waiter_pos_key:
                                        keys_to_remove.append(key)
                                
                                # Keep only the last 5 positions to prevent memory issues
                                if len(keys_to_remove) > 5:
                                    for key in keys_to_remove[:-5]:
                                        del self.previous_waiter_states[key]
            
            # Her durumda görselleştirme (frame varsa)
            if 'frame' in locals() and frame is not None:
                # QR kodları çiz
                if qr_codes:
                    frame = self.draw_qr_codes(frame, qr_codes)
                
                # Yemek tespitlerini çiz (eğer varsa)
                if 'detected_foods' in locals() and detected_foods:
                    plates_to_draw = plates if 'plates' in locals() else None
                    frame = self.food_detector.draw_food_detections(frame, detected_foods, plates_to_draw)
                
                # Masa durumlarını çiz
                frame = self.draw_table_status(frame)
                
                # Masa hesabını çiz (MASA_1 için)
                frame = self.food_detector.draw_table_bill(frame, 'MASA_1', (frame.shape[1] - 300, 50))
                
                # Video bilgilerini çiz (yeni boyuta göre ayarlanmış)
                info_text = f"Frame: {frame_count}/{frame_count_total} | {frame_count/fps:.1f}s/{duration:.1f}s"
                if paused:
                    info_text += " | DURAKLADI"
                
                cv2.putText(frame, info_text, (10, new_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # QR kod sayısını göster
                qr_count_text = f"QR Codes: {len(qr_codes) if qr_codes else 0}"
                cv2.putText(frame, qr_count_text, (new_width - 150, new_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Enhanced waiter tracking görselleştirmesi kaldırıldı (çok karmaşa yapıyor)
                
                # Kontrol bilgilerini göster
                control_text = "ESC:Cikis SPACE:Duraklat R:Baslat C:Hesap_Sifirla"
                cv2.putText(frame, control_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Ekranda göster
                cv2.imshow(window_name, frame)
            
            # Klavye kontrolü - daha düşük bekleme süresi
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - Çıkış
                print("\n👋 Sistem kapatılıyor...")
                break
            elif key == ord(' '):  # SPACE - Duraklat/Devam et
                paused = not paused
                print(f"⏸️ {'Duraklatıldı' if paused else '▶️ Devam ediliyor'}")
                if not paused:
                    last_frame_time = time.time()  # Zamanlayıcıyı sıfırla
            elif key == ord('r') or key == ord('R'):  # R - Başa dön
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                paused = False
                last_frame_time = time.time()
                print("🔄 Video başa döndürüldü")
            elif key == ord('q') or key == ord('Q'):  # Q - Hızlı çıkış
                break
            elif key == ord('f') or key == ord('F'):  # F - Tam ekran toggle
                # Pencere modunu değiştir
                cv2.destroyWindow(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, new_width, new_height)
                print("🖥️ Pencere boyutu yenilendi")
            elif key == ord('c') or key == ord('C'):  # C - Hesap sıfırla
                # MASA_1 hesabını manuel sıfırla
                old_total = self.food_detector.clear_table_bill('MASA_1')
                print(f"🧾 MASA_1 hesabı manuel olarak sıfırlandı (Önceki: {old_total:.0f} TL)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Son durum raporu
        print(f"\n📊 Son Durum Raporu:")
        
        # Masa durumları
        print("📋 Masa Durumları:")
        status_list = self.table_manager.get_table_status_display()
        for status in status_list:
            table = status['table']
            table_status = status['status']
            customer_count = status['customer_count']
            
            if table_status == "empty":
                status_emoji = "✅"
            elif table_status == "waiting":
                status_emoji = "⏰"
            elif table_status == "served":
                status_emoji = "🍽️"
            else:
                status_emoji = "👥"
            
            print(f"   {table}: {table_status.upper()} {status_emoji} (Müşteri sayısı: {customer_count})")
        
        # Garson performansı
        print("\n👨‍💼 Garson Performansı:")
        performance = self.table_manager.get_performance_summary()
        for waiter_id, perf in performance.items():
            print(f"   {waiter_id}:")
            print(f"     • Performans Skoru: {perf['performance_score']}/100")
            print(f"     • Ortalama Yanıt: {perf['avg_response']}s")
            print(f"     • Toplam Servis: {perf['total_services']}")
            print(f"     • Uyarı Sayısı: {perf['warnings']}")
        
        # Yemek tespiti raporu
        print("\n🍽️ Yemek Tespiti Raporu:")
        food_summaries = self.food_detector.get_all_tables_summary()
        if food_summaries:
            for summary in food_summaries:
                table_id = summary['table_id']
                items = summary['items']
                total_price = summary['total_price']
                print(f"   {table_id}:")
                print(f"     • Toplam Tutar: {total_price:.0f} TL")
                print(f"     • Yemek Sayısı: {summary['total_items']}")
                if items:
                    print("     • Siparişler:")
                    for food_name, count in items.items():
                        print(f"       - {count}x {food_name}")
        else:
            print("   Henüz yemek tespiti yapılmadı.")
        
        return self.table_states

    def _translate_qr_code(self, qr_data):
        """
        Demo video QR kodlarını standart formata çevir
        """
        return self.qr_translation.get(qr_data, qr_data)
    
    def _is_table_qr(self, qr_data):
        """
        QR kodun masa kodu olup olmadığını kontrol et
        """
        translated = self._translate_qr_code(qr_data)
        return translated.startswith("TABLE_") or qr_data in ["m001", "m002", "m003", "m004"]
    
    def _is_waiter_qr(self, qr_data):
        """
        QR kodun garson kodu olup olmadığını kontrol et
        """
        translated = self._translate_qr_code(qr_data)
        return translated.startswith("WAITER_") or qr_data in ["w001", "w002", "g001", "g002"]
    
# Test fonksiyonu
def test_qr_detector():
    """
    QR tespit sistemini test et
    """
    detector = QRCodeDetector()
    
    print("🧪 QR Kod Tespit Sistemi Test Ediliyor...")
    print("📝 Masa durumları:")
    
    for table, state in detector.table_states.items():
        print(f"   {table}: {state['status']}")
    
    print("\n🎯 Test tamamlandı!")
    
    # Eğer test videosu varsa işle
    test_video = "test_video.mp4"  # Bu dosyayı sonra ekleyeceğiz
    
    print(f"\n📹 Test videosu aranıyor: {test_video}")
    print("💡 Test videosu yoksa, önce örnek video dosyası eklemeniz gerekir.")
    
    return detector

if __name__ == "__main__":
    # Test çalıştır
    detector = test_qr_detector()
    
    # Kullanıcıdan video seçimi iste
    print("\n" + "="*60)
    print("🎬 DEMO VIDEO SECIMI")
    print("="*60)
    print("Hangi demo videoyu kullanmak istiyorsunuz?")
    print("1. demo_video.mp4 (Birinci aci)")
    print("2. demo_video2.mp4 (Ikinci aci)")
    print("3. Kendi video dosyanizi belirtin")
    
    while True:
        choice = input("\nSeciminizi yapin (1/2/3): ").strip()
        
        if choice == "1":
            video_file = "demo/demo_video.mp4"
            print(f"✅ Secildi: {video_file}")
            break
        elif choice == "2":
            video_file = "demo/demo_video2.mp4"
            print(f"✅ Secildi: {video_file}")
            break
        elif choice == "3":
            print("\n📁 Kendi video dosyanizi belirtin:")
            print("Video dosyanizi proje klasorune koyun ve adini girin:")
            print("Onerilen format: .mp4, .avi, .mov")
            print("Ornek: demo/my_video.mp4")
            
            video_file = input("\n📹 Video dosya adi: ").strip()
            
            if not video_file:
                print("❌ Lutfen bir dosya adi girin!")
                continue
                
            # Dosya uzantısı kontrolü
            if not any(video_file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                print("⚠️ Video dosya uzantisi eklemeyi unutmayin (.mp4, .avi, vb.)")
                continue
            
            print(f"✅ Secildi: {video_file}")
            break
        else:
            print("❌ Lutfen 1, 2 veya 3 secin!")
            continue
    
    # Video dosyasının varlığını kontrol et
    import os
    if not os.path.exists(video_file):
        print(f"\n❌ Video dosyasi bulunamadi: {video_file}")
        print("💡 Lutfen dosyanin dogru konumda oldugunu emin olun.")
        exit(1)
    
    try:
        print(f"\n🚀 Video isleme baslatiliyor: {video_file}")
        print("💡 Kontroller:")
        print("   [ESC] - Cikis")
        print("   [SPACE] - Duraklat/Devam et") 
        print("   [R] - Basa don")
        print("   [Q] - Hizli cikis")
        print("   [F] - Pencere boyutunu yenile")
        print("   [C] - Hesap sifirla")
        print("\n⚠️ Video cok buyukse otomatik olarak kucultulecek")
        print("⏳ Video aciliyor...")
        
        final_states = detector.process_video(video_file)
        
        if final_states:
            print(f"\n✅ Video isleme tamamlandi!")
            print(f"🎯 Sistem hazir, video dosyasi basariyla islendi.")
        else:
            print(f"\n❌ Video islenemedi!")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Kullanici tarafindan durduruldu")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        print("🔄 Lutfen video dosyasinin dogru formatta oldugunu emin olun.")
