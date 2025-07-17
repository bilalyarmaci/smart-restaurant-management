"""
Adım 2: Müşteri Geldiğinde Zamanlayıcı Başlatma
TableManager sınıfı - Masa durumları ve zamanlayıcı yönetimi
"""

from datetime import datetime, timedelta
from enum import Enum
import time

class TableStatus(Enum):
    """Masa durumları"""
    EMPTY = "empty"          # Boş masa (QR kod görünür)
    OCCUPIED = "occupied"    # Müşteri var (QR kod görünmez)
    WAITING = "waiting"      # Müşteri geldi, garson bekleniyor
    SERVED = "served"        # Garson geldi, servis yapıldı

class TableTimer:
    """Her masa için zamanlayıcı"""
    def __init__(self, table_id):
        self.table_id = table_id
        self.customer_arrival_time = None
        self.waiter_arrival_time = None
        self.service_start_time = None
        self.response_time = None
        self.warning_issued = False
        
    def start_customer_timer(self):
        """Müşteri geldiğinde zamanlayıcıyı başlat"""
        self.customer_arrival_time = datetime.now()
        self.waiter_arrival_time = None
        self.service_start_time = None
        self.response_time = None
        self.warning_issued = False
        print(f"⏰ {self.table_id.upper()}: Müşteri zamanlayıcısı başlatıldı ({self.customer_arrival_time.strftime('%H:%M:%S')})")
    
    def waiter_arrived(self):
        """Garson geldiğinde zamanlayıcıyı durdur"""
        if self.customer_arrival_time:
            self.waiter_arrival_time = datetime.now()
            self.response_time = (self.waiter_arrival_time - self.customer_arrival_time).total_seconds()
            print(f"👨‍💼 {self.table_id.upper()}: Garson geldi! Yanıt süresi: {self.response_time:.1f} saniye")
            return self.response_time
        return None
    
    def get_waiting_time(self):
        """Şu anki bekleme süresini al"""
        if self.customer_arrival_time and not self.waiter_arrival_time:
            return (datetime.now() - self.customer_arrival_time).total_seconds()
        return 0
    
    def check_warning(self, warning_threshold=60):
        """Uyarı kontrolü (varsayılan 60 saniye)"""
        waiting_time = self.get_waiting_time()
        if waiting_time > warning_threshold and not self.warning_issued:
            self.warning_issued = True
            print(f"⚠️ UYARI: {self.table_id.upper()} - Garson {waiting_time:.1f} saniyedir gelmedi!")
            return True
        return False
    
    def reset(self):
        """Zamanlayıcıyı sıfırla"""
        self.customer_arrival_time = None
        self.waiter_arrival_time = None
        self.service_start_time = None
        self.response_time = None
        self.warning_issued = False

class TableManager:
    """Masa yönetimi sistemi"""
    def __init__(self):
        self.tables = {
            "table_1": {
                "status": TableStatus.EMPTY,
                "qr_visible": True,
                "last_update": None,
                "timer": TableTimer("table_1"),
                "waiter_assigned": None,  # hangi garson sorumlu
                "customer_count": 0,
                "total_waiting_time": 0,
                "service_count": 0
            },
            "table_2": {
                "status": TableStatus.EMPTY,
                "qr_visible": True,
                "last_update": None,
                "timer": TableTimer("table_2"),
                "waiter_assigned": None,
                "customer_count": 0,
                "total_waiting_time": 0,
                "service_count": 0
            },
            "table_3": {
                "status": TableStatus.EMPTY,
                "qr_visible": True,
                "last_update": None,
                "timer": TableTimer("table_3"),
                "waiter_assigned": None,
                "customer_count": 0,
                "total_waiting_time": 0,
                "service_count": 0
            },
            "table_4": {
                "status": TableStatus.EMPTY,
                "qr_visible": True,
                "last_update": None,
                "timer": TableTimer("table_4"),
                "waiter_assigned": None,
                "customer_count": 0,
                "total_waiting_time": 0,
                "service_count": 0
            }
        }
        
        # Garson-masa atamaları (proje tanımına göre)
        self.waiter_assignments = {
            "table_1": "GARSON_1",  # İlk iki masa birinci garson
            "table_2": "GARSON_1",
            "table_3": "GARSON_2",  # Son iki masa ikinci garson
            "table_4": "GARSON_2"
        }
        
        # Garson performans takibi
        self.waiter_performance = {
            "GARSON_1": {
                "total_responses": 0,
                "total_response_time": 0,
                "average_response_time": 0,
                "warnings": 0,
                "tables_served": []
            },
            "GARSON_2": {
                "total_responses": 0,
                "total_response_time": 0,
                "average_response_time": 0,
                "warnings": 0,
                "tables_served": []
            }
        }
    
    def update_table_qr_status(self, table_qr_codes):
        """QR kod durumlarına göre masa durumlarını güncelle"""
        current_time = datetime.now()
        
        # Hangi masa QR kodları görüldü
        visible_tables = []
        for qr_code in table_qr_codes:
            if qr_code.startswith("MASA_"):
                table_num = qr_code.split('_')[1]
                table_name = f"table_{table_num}"
                visible_tables.append(table_name)
        
        # Tüm masaları kontrol et
        for table_name, table_data in self.tables.items():
            
            if table_name in visible_tables:
                # QR kod görülüyor - masa boş veya müşteri kalktı
                if not table_data["qr_visible"]:
                    print(f"📋 {table_name.upper()}: Müşteri kalktı, masa boşaldı!")
                    self._customer_left(table_name)
                
                table_data["qr_visible"] = True
                table_data["status"] = TableStatus.EMPTY
                table_data["last_update"] = current_time
                
            else:
                # QR kod görülmüyor - müşteri geldi
                if table_data["qr_visible"]:
                    print(f"👥 {table_name.upper()}: Müşteri geldi!")
                    self._customer_arrived(table_name)
                
                table_data["qr_visible"] = False
                if table_data["status"] == TableStatus.EMPTY:
                    table_data["status"] = TableStatus.WAITING
                table_data["last_update"] = current_time
    
    def _customer_arrived(self, table_name):
        """Müşteri geldiğinde"""
        table_data = self.tables[table_name]
        table_data["status"] = TableStatus.WAITING
        table_data["timer"].start_customer_timer()
        table_data["customer_count"] += 1
        
        # Garson ataması
        assigned_waiter = self.waiter_assignments[table_name]
        table_data["waiter_assigned"] = assigned_waiter
        
        print(f"🔔 {table_name.upper()}: Atanan garson: {assigned_waiter}")
    
    def _customer_left(self, table_name):
        """Müşteri kalktığında"""
        table_data = self.tables[table_name]
        
        # Eğer garson gelmemişse, bekleme süresini kaydet
        if table_data["status"] == TableStatus.WAITING:
            waiting_time = table_data["timer"].get_waiting_time()
            table_data["total_waiting_time"] += waiting_time
            
            # Garson performansına olumsuz kayıt - SADECE 60+ saniye için
            assigned_waiter = table_data["waiter_assigned"]
            if assigned_waiter and waiting_time >= 60.0:  # 60 saniye ve üzeri için eksi puan
                self.waiter_performance[assigned_waiter]["warnings"] += 1
                print(f"⚠️ {assigned_waiter}: Performans puanı düştü (müşteri {waiting_time:.1f}s bekledi)")
            elif assigned_waiter and waiting_time < 60.0:
                print(f"✅ {assigned_waiter}: Müşteri {waiting_time:.1f}s bekledi (60s altında - puan düşmedi)")
        
        # Masa durumunu sıfırla
        table_data["status"] = TableStatus.EMPTY
        table_data["timer"].reset()
        table_data["waiter_assigned"] = None
    
    def waiter_detected(self, waiter_id, table_position=None):
        """Garson tespit edildiğinde"""
        # Hangi masaya yakın olduğunu belirle
        target_table = None
        
        # Eğer pozisyon verilmişse, en yakın masayı bul
        if table_position:
            # Şimdilik basit logic - ilerleyen adımlarda geliştirilecek
            # Garson atama sistemine göre hangi masaları kontrol etmeli
            for table_name, assigned_waiter in self.waiter_assignments.items():
                if assigned_waiter == waiter_id:
                    table_data = self.tables[table_name]
                    if table_data["status"] == TableStatus.WAITING:
                        target_table = table_name
                        break
        
        if target_table:
            response_time = self.tables[target_table]["timer"].waiter_arrived()
            self.tables[target_table]["status"] = TableStatus.SERVED
            self.tables[target_table]["service_count"] += 1
            
            # Garson performansını güncelle
            self._update_waiter_performance(waiter_id, response_time, target_table)
            
            return target_table, response_time
        
        return None, None
    
    def _update_waiter_performance(self, waiter_id, response_time, table_name):
        """Garson performansını güncelle"""
        if waiter_id in self.waiter_performance and response_time:
            perf = self.waiter_performance[waiter_id]
            perf["total_responses"] += 1
            perf["total_response_time"] += response_time
            perf["average_response_time"] = perf["total_response_time"] / perf["total_responses"]
            perf["tables_served"].append({
                "table": table_name,
                "response_time": response_time,
                "timestamp": datetime.now()
            })
    
    def check_warnings(self, warning_threshold=60):
        """Tüm masalar için uyarı kontrolü"""
        warnings = []
        for table_name, table_data in self.tables.items():
            if table_data["status"] == TableStatus.WAITING:
                if table_data["timer"].check_warning(warning_threshold):
                    warnings.append({
                        "table": table_name,
                        "waiter": table_data["waiter_assigned"],
                        "waiting_time": table_data["timer"].get_waiting_time()
                    })
        return warnings
    
    def get_performance_summary(self):
        """Performans özetini al"""
        summary = {}
        for waiter_id, perf in self.waiter_performance.items():
            summary[waiter_id] = {
                "avg_response": round(perf["average_response_time"], 1),
                "total_services": perf["total_responses"],
                "warnings": perf["warnings"],
                "performance_score": self._calculate_performance_score(waiter_id)
            }
        return summary
    
    def _calculate_performance_score(self, waiter_id):
        """Garson performans skoru hesapla (0-100)"""
        perf = self.waiter_performance[waiter_id]
        
        if perf["total_responses"] == 0:
            return 100  # Henüz servis yapmamış
        
        # Temel skor: 100
        score = 100
        
        # Ortalama yanıt süresine göre puan kaybı
        avg_time = perf["average_response_time"]
        if avg_time > 60:  # 60 saniyeden fazla
            score -= min(50, (avg_time - 60) / 2)  # Maksimum 50 puan kaybı
        
        # Uyarı sayısına göre puan kaybı
        score -= perf["warnings"] * 10  # Her uyarı için 10 puan kaybı
        
        return max(0, round(score, 1))
    
    def get_table_status_display(self):
        """Masa durumlarını görüntüleme için formatla"""
        status_list = []
        for table_name, table_data in self.tables.items():
            status_info = {
                "table": table_name.replace("table_", "MASA_"),
                "status": table_data["status"].value,
                "qr_visible": table_data["qr_visible"],
                "waiting_time": table_data["timer"].get_waiting_time() if table_data["status"] == TableStatus.WAITING else 0,
                "assigned_waiter": table_data["waiter_assigned"],
                "customer_count": table_data["customer_count"]
            }
            status_list.append(status_info)
        return status_list

# Test fonksiyonu
def test_table_manager():
    """TableManager sistemini test et"""
    print("🧪 TableManager Test Ediliyor...")
    
    manager = TableManager()
    
    # Başlangıç durumu
    print("\n📋 Başlangıç Masa Durumları:")
    for status in manager.get_table_status_display():
        print(f"   {status['table']}: {status['status']}")
    
    # Test senaryosu
    print("\n🎬 Test Senaryosu:")
    print("1. Table_1'e müşteri geliyor...")
    manager.update_table_qr_status([])  # QR kod görünmüyor
    
    time.sleep(2)
    
    print("2. 2 saniye sonra garson geliyor...")
    manager.waiter_detected("WAITER_1")
    
    print("\n📊 Performans Özeti:")
    summary = manager.get_performance_summary()
    for waiter, perf in summary.items():
        print(f"   {waiter}: Skor: {perf['performance_score']}, Ortalama: {perf['avg_response']}s")

if __name__ == "__main__":
    test_table_manager()
