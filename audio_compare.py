import numpy as np
import cv2
import librosa
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
import threading

class ShinyDetector:
    def __init__(self, method='hybrid', **kwargs):
        """
        Initialize shiny detector with multiple detection methods.
        
        Args:
            method: 'audio_simple', 'visual', or 'hybrid'
        """
        self.method = method
        
        if method in ['audio_simple', 'hybrid']:
            self.setup_audio_detection(**kwargs)
        if method in ['visual', 'hybrid']:
            self.setup_visual_detection(**kwargs)
    
    def setup_audio_detection(self, shiny_jingle_path='shiny_jingle.mp3', sr=22050, **kwargs):
        """Setup simple cross-correlation audio detection."""
        print("Setting up simple audio detection...")
        self.sr = sr
        
        # Load reference jingle
        self.reference_audio, _ = librosa.load(shiny_jingle_path, sr=self.sr)
        print(f"Loaded reference: {len(self.reference_audio)} samples, {len(self.reference_audio)/self.sr:.2f}s")
        
        # Normalize reference audio
        self.reference_audio = self.reference_audio / np.max(np.abs(self.reference_audio))
        
    def setup_visual_detection(self, **kwargs):
        """Setup visual detection for shiny effects."""
        print("Setting up visual detection...")
        
        # Common shiny colors (HSV ranges)
        self.shiny_colors = {
            'gold': [(15, 100, 100), (35, 255, 255)],      # Golden sparkles
            'silver': [(0, 0, 180), (180, 30, 255)],       # Silver/white sparkles  
            'stars': [(0, 0, 200), (180, 50, 255)],        # Bright white stars
        }
        
        # Template for star/sparkle shapes (will be generated)
        self.star_templates = self._create_star_templates()
    
    def _create_star_templates(self):
        """Create templates for star/sparkle detection."""
        templates = []
        
        # Create different sized star templates
        for size in [8, 12, 16, 20]:
            template = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            
            # Create a simple star pattern
            template[center, :] = 255  # Horizontal line
            template[:, center] = 255  # Vertical line
            template[center-1:center+2, center-1:center+2] = 255  # Center blob
            
            templates.append(template)
            
        return templates
    
    def detect_audio_simple(self, audio_path: str) -> Tuple[bool, List[dict]]:
        """Simple cross-correlation audio detection."""
        print(f"Analyzing audio: {audio_path}")
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr)
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        print(f"Audio length: {len(audio)/self.sr:.2f}s")
        
        # Simple cross-correlation
        correlation = signal.correlate(audio, self.reference_audio, mode='valid')
        correlation = np.abs(correlation)  # Take magnitude
        
        # Normalize correlation
        correlation = correlation / np.max(correlation)
        
        # Find peaks
        threshold = 0.7  # Adjust as needed
        peaks, properties = signal.find_peaks(correlation, height=threshold, distance=len(self.reference_audio)//2)
        
        # Convert to time and create detections
        detections = []
        for peak in peaks:
            timestamp = peak / self.sr
            confidence = correlation[peak]
            detections.append({
                'timestamp': timestamp,
                'confidence': confidence,
                'method': 'cross_correlation'
            })
        
        # Plot results
        plt.figure(figsize=(15, 8))
        
        # Plot original audio
        plt.subplot(3, 1, 1)
        time_audio = np.linspace(0, len(audio)/self.sr, len(audio))
        plt.plot(time_audio, audio)
        plt.title('Original Audio')
        plt.ylabel('Amplitude')
        
        # Plot reference
        plt.subplot(3, 1, 2)
        time_ref = np.linspace(0, len(self.reference_audio)/self.sr, len(self.reference_audio))
        plt.plot(time_ref, self.reference_audio)
        plt.title('Reference Jingle')
        plt.ylabel('Amplitude')
        
        # Plot correlation
        plt.subplot(3, 1, 3)
        time_corr = np.linspace(0, len(correlation)/self.sr, len(correlation))
        plt.plot(time_corr, correlation, label='Cross-correlation')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        
        # Mark detections
        for detection in detections:
            plt.axvline(x=detection['timestamp'], color='red', alpha=0.8, 
                       label=f"Detection ({detection['confidence']:.3f})")
        
        plt.title('Cross-correlation Results')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Correlation')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        detection_found = len(detections) > 0
        return detection_found, detections
    
    def detect_visual_sparkles(self, image_path: str) -> Tuple[bool, List[dict]]:
        """Detect visual sparkles/stars in an image."""
        print(f"Analyzing image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return False, []
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        detections = []
        
        # Check for bright sparkle colors
        for color_name, (lower, upper) in self.shiny_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'type': 'color_detection',
                        'color': color_name,
                        'position': (x + w//2, y + h//2),
                        'area': area,
                        'confidence': min(area / 100, 1.0)
                    })
        
        # Template matching for stars
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for i, template in enumerate(self.star_templates):
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)
            
            for pt in zip(*locations[::-1]):
                detections.append({
                    'type': 'template_match',
                    'template_size': template.shape[0],
                    'position': pt,
                    'confidence': result[pt[1], pt[0]]
                })
        
        # Visualize results
        result_img = img.copy()
        for detection in detections:
            x, y = detection['position']
            cv2.circle(result_img, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(result_img, f"{detection['confidence']:.2f}", 
                       (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show result
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Detections')
        plt.axis('off')
        plt.show()
        
        detection_found = len(detections) > 0
        return detection_found, detections
    
    def detect_screen_capture(self, region=None):
        """Real-time screen capture and analysis."""
        try:
            import pyautogui
            import PIL
        except ImportError:
            print("Need to install: pip install pyautogui pillow")
            return False, []
        
        # Capture screen
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Use visual detection on screenshot
        return self.detect_visual_sparkles_array(img)
    
    def detect_visual_sparkles_array(self, img: np.ndarray) -> Tuple[bool, List[dict]]:
        """Detect sparkles in a numpy array image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections = []
        
        # Look for bright white/yellow pixels (typical sparkles)
        # High value (brightness) and low saturation (white-ish)
        lower_sparkle = np.array([0, 0, 220])    # Any hue, low saturation, high brightness
        upper_sparkle = np.array([180, 100, 255])
        
        mask = cv2.inRange(hsv, lower_sparkle, upper_sparkle)
        
        # Find bright regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 500:  # Filter appropriate sizes for sparkles
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Sparkles are usually somewhat round
                if 0.5 < aspect_ratio < 2.0:
                    detections.append({
                        'type': 'sparkle',
                        'position': (x + w//2, y + h//2),
                        'area': area,
                        'confidence': min(area / 50, 1.0)
                    })
        
        return len(detections) > 0, detections

# Test different approaches
def test_all_methods():
    """Test both audio and visual detection methods."""
    
    print("="*60)
    print("TESTING SIMPLE AUDIO DETECTION")
    print("="*60)
    
    # Test simple audio detection
    detector = ShinyDetector(method='audio_simple', shiny_jingle_path='shiny_jingle.mp3')
    audio_found, audio_detections = detector.detect_audio_simple('example_encounter2.wav')
    
    print(f"Audio Detection Results:")
    print(f"Found: {audio_found}")
    print(f"Detections: {len(audio_detections)}")
    for det in audio_detections:
        print(f"  Time: {det['timestamp']:.2f}s, Confidence: {det['confidence']:.3f}")
    
    print("\n" + "="*60)
    print("VISUAL DETECTION SETUP")
    print("="*60)
    
    # Setup visual detector
    visual_detector = ShinyDetector(method='visual')
    
    print("Visual detection ready!")
    print("To test visual detection:")
    print("1. Take a screenshot when a shiny appears")
    print("2. Call: visual_detector.detect_visual_sparkles('screenshot.png')")
    print("3. Or use: visual_detector.detect_screen_capture(region=(x,y,w,h))")
    
    return detector, visual_detector

def real_time_monitor_example():
    """Example of real-time monitoring."""
    print("Real-time monitoring example:")
    print("This would capture your game screen and look for sparkles")
    print("Use detect_screen_capture() in a loop during encounters")

if __name__ == "__main__":
    audio_detector, visual_detector = test_all_methods()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("Based on results, I recommend:")
    print("1. If audio detection failed: Try visual detection")
    print("2. Visual detection often works better for shinies")
    print("3. Capture screenshots during encounters and test visual detection")
    print("4. Consider hybrid approach: audio trigger + visual confirmation")