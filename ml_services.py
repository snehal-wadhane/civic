# ml_services.py - Machine Learning Services
import numpy as np
import cv2
from ultralytics import YOLO
from typing import Dict, List, Tuple
import logging
from config import settings

logger = logging.getLogger(__name__)

class IssueDetector:
    """Multi-issue type detection service"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all ML models"""
        try:
            # Pothole detection model (YOLOv8)
            self.models['pothole'] = YOLO(settings.YOLO_MODEL_PATH)
            logger.info("Pothole detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pothole model: {e}")
    
    def detect_pothole(self, image: np.ndarray, min_conf: float = None) -> Dict:
        """
        Detect potholes in image
        
        Args:
            image: OpenCV image (numpy array)
            min_conf: Minimum confidence threshold
            
        Returns:
            dict: {
                "detected": bool,
                "severity": str,
                "confidence": float,
                "count": int,
                "detections": list
            }
        """
        if min_conf is None:
            min_conf = settings.MIN_CONFIDENCE
        
        class_names = ["minor_pothole", "medium_pothole", "major_pothole"]
        
        try:
            model = self.models.get('pothole')
            if not model:
                return self._no_detection_response()
            
            results = model(image)
            res = results[0]
            
            if len(res.boxes) == 0 or res.boxes.conf.max() < min_conf:
                return self._no_detection_response()
            
            cls_indices = res.boxes.cls.cpu().numpy()
            conf_scores = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            
            # Filter by confidence
            valid_indices = conf_scores >= min_conf
            cls_indices = cls_indices[valid_indices]
            conf_scores = conf_scores[valid_indices]
            boxes = boxes[valid_indices]
            
            if len(cls_indices) == 0:
                return self._no_detection_response()
            
            # Count detections per class
            cls_counts = {name: 0 for name in class_names}
            for c in cls_indices:
                cls_counts[class_names[int(c)]] += 1
            
            # Determine dominant class
            image_category = max(cls_counts, key=cls_counts.get)
            indices_of_category = [i for i, c in enumerate(cls_indices) 
                                  if class_names[int(c)] == image_category]
            avg_confidence = float(np.mean(conf_scores[indices_of_category]))
            
            # Prepare detections list
            detections = []
            for i, (box, conf, cls) in enumerate(zip(boxes, conf_scores, cls_indices)):
                detections.append({
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class": class_names[int(cls)]
                })
            
            return {
                "detected": True,
                "severity": image_category,
                "confidence": avg_confidence,
                "count": len(cls_indices),
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Pothole detection error: {e}")
            return self._no_detection_response()
    
    def detect_streetlight(self, image: np.ndarray) -> Dict:
        """
        Detect broken streetlights (placeholder - needs specific model)
        
        For now, uses image analysis heuristics:
        - Check for dark regions in expected light positions
        - Analyze brightness patterns
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Simple heuristic: if image is very dark, might be broken light
            # This is a placeholder - replace with actual ML model
            if mean_brightness < 50:
                confidence = min(0.6 + (50 - mean_brightness) / 100, 0.9)
                return {
                    "detected": True,
                    "severity": "broken_streetlight",
                    "confidence": float(confidence),
                    "count": 1,
                    "detections": [{
                        "type": "low_illumination",
                        "confidence": float(confidence),
                        "brightness": float(mean_brightness)
                    }]
                }
            
            return self._no_detection_response()
            
        except Exception as e:
            logger.error(f"Streetlight detection error: {e}")
            return self._no_detection_response()
    
    def detect_water_leak(self, image: np.ndarray) -> Dict:
        """
        Detect water leaks (placeholder - needs specific model)
        
        Uses color analysis to detect water:
        - Blue/cyan regions
        - Wet surface patterns
        """
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define range for blue/water colors
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for water-like colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Calculate percentage of water-like pixels
            water_percentage = (np.sum(mask > 0) / mask.size) * 100
            
            if water_percentage > 5:  # If >5% of image is water-like
                confidence = min(0.5 + water_percentage / 50, 0.85)
                return {
                    "detected": True,
                    "severity": "water_leak",
                    "confidence": float(confidence),
                    "count": 1,
                    "detections": [{
                        "type": "water_presence",
                        "confidence": float(confidence),
                        "coverage_percentage": float(water_percentage)
                    }]
                }
            
            return self._no_detection_response()
            
        except Exception as e:
            logger.error(f"Water leak detection error: {e}")
            return self._no_detection_response()
    
    def detect_garbage(self, image: np.ndarray) -> Dict:
        """
        Detect garbage accumulation (placeholder - needs specific model)
        """
        try:
            # Placeholder implementation
            # In production, use trained model for garbage detection
            
            # For now, return no detection
            # This should be replaced with actual YOLO model trained on garbage
            return self._no_detection_response()
            
        except Exception as e:
            logger.error(f"Garbage detection error: {e}")
            return self._no_detection_response()
    
    def auto_detect_issue_type(self, image: np.ndarray) -> Tuple[str, Dict]:
        """
        Automatically detect issue type from image
        Tries multiple detectors and returns best match
        
        Returns:
            tuple: (issue_type_name, detection_result)
        """
        detectors = {
            'Pothole': self.detect_pothole,
            'Water Leak': self.detect_water_leak,
            'Broken Streetlight': self.detect_streetlight,
            'Garbage Accumulation': self.detect_garbage,
        }
        
        best_match = None
        best_confidence = 0.0
        
        for issue_type, detector in detectors.items():
            result = detector(image)
            if result['detected'] and result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_match = (issue_type, result)
        
        if best_match:
            return best_match
        
        # If no detection, return unknown
        return ("Unknown", self._no_detection_response())
    
    def _no_detection_response(self) -> Dict:
        """Standard no detection response"""
        return {
            "detected": False,
            "severity": "unknown",
            "confidence": 0.0,
            "count": 0,
            "detections": []
        }

# Global detector instance
detector = IssueDetector()

def analyze_issue_image(image_bytes: bytes, issue_type_name: str = None) -> Dict:
    """
    Analyze image for issue detection
    
    Args:
        image_bytes: Image as bytes
        issue_type_name: Specific issue type to detect (optional)
        
    Returns:
        dict: Detection results
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # If specific issue type given, use that detector
        if issue_type_name:
            if issue_type_name.lower() in ['pothole', 'road damage']:
                return detector.detect_pothole(img)
            elif 'streetlight' in issue_type_name.lower():
                return detector.detect_streetlight(img)
            elif 'water' in issue_type_name.lower() or 'leak' in issue_type_name.lower():
                return detector.detect_water_leak(img)
            elif 'garbage' in issue_type_name.lower():
                return detector.detect_garbage(img)
        
        # Auto-detect issue type
        issue_type, result = detector.auto_detect_issue_type(img)
        result['detected_issue_type'] = issue_type
        return result
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return detector._no_detection_response()
