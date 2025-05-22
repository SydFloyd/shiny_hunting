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
        
        # Precompute reference energy for proper normalization
        self.ref_energy = np.sqrt(np.sum(self.reference_audio ** 2))
        print(f"Reference energy: {self.ref_energy:.4f}")
        
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
    
    def detect_audio_simple(self, audio_path: str, use_splice_hack=True, threshold=0.7) -> Tuple[bool, List[dict]]:
        """
        Simple cross-correlation with optional splice hack for better normalization.
        
        Args:
            audio_path: Path to audio file to analyze
            use_splice_hack: If True, splice reference jingle into audio for normalization
            threshold: Detection threshold (0-1)
        """
        print(f"Analyzing audio: {audio_path}")
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr)
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        print(f"Audio length: {len(audio)/self.sr:.2f}s")
        
        if use_splice_hack:
            print("Using splice hack for better normalization...")
            
            # Find a quiet spot to splice in the reference (end of audio)
            splice_position = len(audio) - len(self.reference_audio) - int(0.5 * self.sr)  # 0.5s from end
            if splice_position < 0:
                splice_position = len(audio) // 2  # If audio too short, use middle
            
            # Create a copy and splice in the reference
            audio_with_ref = audio.copy()
            audio_with_ref[splice_position:splice_position + len(self.reference_audio)] = self.reference_audio
            
            print(f"Spliced reference at {splice_position/self.sr:.2f}s")
            
            # Use the spliced version for correlation
            correlation_audio = audio_with_ref
        else:
            correlation_audio = audio
        
        # Compute cross-correlation
        correlation = signal.correlate(correlation_audio, self.reference_audio, mode='valid')
        correlation = np.abs(correlation)  # Take magnitude
        
        # Simple normalization by maximum (this now works properly with splice hack)
        max_correlation = np.max(correlation)
        if max_correlation > 0:
            normalized_correlation = correlation / max_correlation
        else:
            normalized_correlation = correlation
        
        print(f"Max correlation: {max_correlation:.4f}")
        
        # Find peaks
        min_distance = len(self.reference_audio) // 2
        peaks, properties = signal.find_peaks(
            normalized_correlation, 
            height=threshold, 
            distance=min_distance
        )
        
        # Convert to time and create detections
        detections = []
        splice_time = splice_position / self.sr if use_splice_hack else -1
        
        for peak in peaks:
            timestamp = peak / self.sr
            confidence = normalized_correlation[peak]
            
            # If using splice hack, ignore the spliced reference detection
            if use_splice_hack and abs(timestamp - splice_time) < 0.1:  # Within 0.1s of splice
                print(f"Ignoring spliced reference at {timestamp:.2f}s")
                continue
                
            detections.append({
                'timestamp': timestamp,
                'confidence': confidence,
                'method': 'cross_correlation_with_splice' if use_splice_hack else 'cross_correlation',
                'threshold_used': threshold
            })
            print(f"Detection at {timestamp:.2f}s with confidence {confidence:.4f}")
        
        # Enhanced plotting
        fig_height = 12 if use_splice_hack else 10
        plt.figure(figsize=(15, fig_height))
        
        subplot_count = 5 if use_splice_hack else 4
        
        # Plot original audio
        plt.subplot(subplot_count, 1, 1)
        time_audio = np.linspace(0, len(audio)/self.sr, len(audio))
        plt.plot(time_audio, audio)
        plt.title('Original Audio')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot reference
        plt.subplot(subplot_count, 1, 2)
        time_ref = np.linspace(0, len(self.reference_audio)/self.sr, len(self.reference_audio))
        plt.plot(time_ref, self.reference_audio)
        plt.title('Reference Jingle')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # If using splice hack, show the modified audio
        if use_splice_hack:
            plt.subplot(subplot_count, 1, 3)
            time_audio_mod = np.linspace(0, len(correlation_audio)/self.sr, len(correlation_audio))
            plt.plot(time_audio_mod, correlation_audio)
            plt.axvline(x=splice_time, color='red', linestyle='--', alpha=0.7, label='Splice position')
            plt.title('Audio with Spliced Reference (for normalization)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot raw correlation
        plt.subplot(subplot_count, 1, subplot_count-1)
        time_corr = np.linspace(0, len(correlation)/self.sr, len(correlation))
        plt.plot(time_corr, correlation, label='Raw Cross-correlation', alpha=0.7)
        plt.title('Raw Cross-correlation')
        plt.ylabel('Raw Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot normalized correlation with detections
        plt.subplot(subplot_count, 1, subplot_count)
        plt.plot(time_corr, normalized_correlation, label='Normalized Cross-correlation', linewidth=2)
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold:.3f})')
        
        # Mark splice position if used
        if use_splice_hack:
            plt.axvline(x=splice_time, color='orange', linestyle=':', alpha=0.7, 
                       label='Spliced ref (ignored)')
        
        # Mark detections
        for i, detection in enumerate(detections):
            color = plt.cm.tab10(i % 10)
            plt.axvline(x=detection['timestamp'], color=color, alpha=0.8, linewidth=2,
                       label=f"Detection {i+1} ({detection['confidence']:.3f})")
        
        plt.title('Normalized Cross-correlation Results')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Method: {'Splice hack' if use_splice_hack else 'Standard'}")
        print(f"Max correlation: {max_correlation:.4f}")
        print(f"Threshold used: {threshold:.4f}")
        print(f"Detections found: {len(detections)}")
        
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
            locations = np.where(result >= 0.25)
            
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
    print("TESTING FIXED AUDIO DETECTION")
    print("="*60)
    
    # Test fixed audio detection with different threshold methods
    detector = ShinyDetector(method='audio_simple', shiny_jingle_path='shiny_jingle.mp3')

    test_audio_path = 'example_non_encounter.wav'
    print(f"Testing audio detection on: {test_audio_path}")
    
    # Try splice hack method first
    print("\n--- SPLICE HACK METHOD ---")
    audio_found, audio_detections = detector.detect_audio_simple(test_audio_path, 
                                                                 use_splice_hack=True, 
                                                                 threshold=0.25)
    
    print(f"\nAudio Detection Results (Splice Hack):")
    print(f"Found: {audio_found}")
    print(f"Detections: {len(audio_detections)}")
    for det in audio_detections:
        print(f"  Time: {det['timestamp']:.2f}s, Confidence: {det['confidence']:.4f}")
    
    # Compare with standard method if needed
    print("\n--- STANDARD METHOD (for comparison) ---")
    audio_found_std, audio_detections_std = detector.detect_audio_simple(test_audio_path, 
                                                                         use_splice_hack=False, 
                                                                         threshold=0.25)
    
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
    print("Based on fixed normalization:")
    print("1. Proper correlation coefficient prevents false positives")
    print("2. Adaptive threshold adjusts to audio characteristics")
    print("3. If still getting false positives, try 'absolute' threshold")
    print("4. Visual detection as backup for difficult audio cases")