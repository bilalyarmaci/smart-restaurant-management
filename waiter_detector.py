"""
Enhanced Waiter Detection System
Tracks waiter movements and proximity to tables
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass
class Position:
    x: int
    y: int
    timestamp: datetime
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class WaiterDetection:
    waiter_id: str
    position: Position
    confidence: float
    qr_data: str

class WaiterTracker:
    def __init__(self):
        self.waiter_positions: Dict[str, List[Position]] = {}
        self.table_positions: Dict[str, Position] = {
            'TABLE_1': Position(164, 346, datetime.now()),  # Based on our detection
            'MASA_2': Position(164, 600, datetime.now()),  # Estimated positions
            'MASA_3': Position(400, 346, datetime.now()),
            'MASA_4': Position(400, 600, datetime.now()),
        }
        self.proximity_threshold = 150  # pixels
        self.waiter_at_table: Dict[str, str] = {}  # waiter_id -> table_id
        
    def add_waiter_detection(self, detection: WaiterDetection):
        """Add a waiter detection to tracking history"""
        waiter_id = detection.waiter_id
        
        if waiter_id not in self.waiter_positions:
            self.waiter_positions[waiter_id] = []
            
        self.waiter_positions[waiter_id].append(detection.position)
        
        # Keep only last 30 positions (memory management)
        if len(self.waiter_positions[waiter_id]) > 30:
            self.waiter_positions[waiter_id] = self.waiter_positions[waiter_id][-30:]
    
    def get_waiter_velocity(self, waiter_id: str) -> float:
        """Calculate waiter movement velocity"""
        if waiter_id not in self.waiter_positions:
            return 0.0
            
        positions = self.waiter_positions[waiter_id]
        if len(positions) < 2:
            return 0.0
            
        recent_positions = positions[-5:]  # Last 5 positions
        if len(recent_positions) < 2:
            return 0.0
            
        total_distance = 0
        for i in range(1, len(recent_positions)):
            total_distance += recent_positions[i].distance_to(recent_positions[i-1])
            
        time_diff = (recent_positions[-1].timestamp - recent_positions[0].timestamp).total_seconds()
        if time_diff == 0:
            return 0.0
            
        return total_distance / time_diff  # pixels per second
    
    def check_table_proximity(self, waiter_id: str) -> Optional[str]:
        """Check if waiter is close to any table"""
        if waiter_id not in self.waiter_positions or not self.waiter_positions[waiter_id]:
            return None
            
        current_position = self.waiter_positions[waiter_id][-1]
        
        for table_id, table_pos in self.table_positions.items():
            distance = current_position.distance_to(table_pos)
            if distance <= self.proximity_threshold:
                return table_id
                
        return None
    
    def update_waiter_table_assignment(self, waiter_id: str, table_id: Optional[str]):
        """Update which table a waiter is currently at"""
        if table_id:
            self.waiter_at_table[waiter_id] = table_id
            print(f"🎯 {waiter_id} is now at {table_id}")
        else:
            if waiter_id in self.waiter_at_table:
                old_table = self.waiter_at_table[waiter_id]
                del self.waiter_at_table[waiter_id]
                print(f"🚶 {waiter_id} left {old_table}")
    
    def get_waiter_status(self, waiter_id: str) -> Dict:
        """Get comprehensive waiter status"""
        status = {
            'id': waiter_id,
            'current_table': self.waiter_at_table.get(waiter_id),
            'velocity': self.get_waiter_velocity(waiter_id),
            'position_history_count': len(self.waiter_positions.get(waiter_id, [])),
            'last_position': None
        }
        
        if waiter_id in self.waiter_positions and self.waiter_positions[waiter_id]:
            last_pos = self.waiter_positions[waiter_id][-1]
            status['last_position'] = (last_pos.x, last_pos.y)
            
        return status
    
    def draw_tracking_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw waiter tracking information on frame"""
        # Draw table positions
        for table_id, pos in self.table_positions.items():
            cv2.circle(frame, (pos.x, pos.y), self.proximity_threshold, (0, 255, 255), 2)
            cv2.putText(frame, table_id, (pos.x - 30, pos.y - 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw waiter positions and trails
        for waiter_id, positions in self.waiter_positions.items():
            if not positions:
                continue
                
            # Draw trail (last 10 positions)
            trail_positions = positions[-10:]
            for i in range(1, len(trail_positions)):
                pt1 = (trail_positions[i-1].x, trail_positions[i-1].y)
                pt2 = (trail_positions[i].x, trail_positions[i].y)
                alpha = i / len(trail_positions)  # Fade effect
                color = (0, int(255 * alpha), 0)
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw current position
            current_pos = positions[-1]
            cv2.circle(frame, (current_pos.x, current_pos.y), 8, (0, 255, 0), -1)
            
            # Draw waiter info
            velocity = self.get_waiter_velocity(waiter_id)
            info_text = f"{waiter_id} v:{velocity:.1f}"
            cv2.putText(frame, info_text, (current_pos.x + 15, current_pos.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

class EnhancedWaiterDetector:
    def __init__(self):
        self.tracker = WaiterTracker()
        self.qr_translation = {
            'w001': 'GARSON_1',
            'g001': 'GARSON_1',  # Demo video uses g001
            'w002': 'GARSON_2',
            'g002': 'GARSON_2'
        }
    
    def process_waiter_qr(self, qr_data: str, position: Tuple[int, int], frame_time: datetime) -> Optional[str]:
        """Process waiter QR detection and return waiter ID"""
        if qr_data not in self.qr_translation:
            return None
            
        waiter_id = self.qr_translation[qr_data]
        
        # Create detection
        detection = WaiterDetection(
            waiter_id=waiter_id,
            position=Position(position[0], position[1], frame_time),
            confidence=1.0,
            qr_data=qr_data
        )
        
        # Add to tracker
        self.tracker.add_waiter_detection(detection)
        
        # Check proximity to tables
        nearby_table = self.tracker.check_table_proximity(waiter_id)
        self.tracker.update_waiter_table_assignment(waiter_id, nearby_table)
        
        return waiter_id
    
    def get_all_waiter_status(self) -> Dict[str, Dict]:
        """Get status of all tracked waiters"""
        status = {}
        for waiter_id in self.tracker.waiter_positions.keys():
            status[waiter_id] = self.tracker.get_waiter_status(waiter_id)
        return status
    
    def draw_enhanced_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced tracking visualization"""
        return self.tracker.draw_tracking_info(frame)

def test_waiter_detector():
    """Test the waiter detection system"""
    detector = EnhancedWaiterDetector()
    
    # Simulate some detections
    import time
    current_time = datetime.now()
    
    print("🧪 Testing Enhanced Waiter Detection")
    print("=" * 50)
    
    # Test waiter movement
    test_positions = [
        (100, 700, 'g001'),  # Start position
        (120, 680, 'g001'),  # Moving towards table
        (140, 660, 'g001'),
        (160, 640, 'g001'),
        (160, 400, 'g001'),  # Near TABLE_1
        (160, 350, 'g001'),  # At TABLE_1
    ]
    
    for i, (x, y, qr_code) in enumerate(test_positions):
        print(f"\n🎯 Detection {i+1}: {qr_code} at ({x}, {y})")
        waiter_id = detector.process_waiter_qr(qr_code, (x, y), current_time)
        
        if waiter_id:
            status = detector.tracker.get_waiter_status(waiter_id)
            print(f"   Waiter: {status['id']}")
            print(f"   Velocity: {status['velocity']:.2f} px/s")
            print(f"   At table: {status['current_table']}")
        
        current_time = datetime.now()
        time.sleep(0.1)  # Simulate time passage
    
    print("\n📊 Final Status:")
    all_status = detector.get_all_waiter_status()
    for waiter_id, status in all_status.items():
        print(f"   {waiter_id}: {status}")

# Backward compatibility alias
WaiterDetector = EnhancedWaiterDetector

if __name__ == "__main__":
    test_waiter_detector()
