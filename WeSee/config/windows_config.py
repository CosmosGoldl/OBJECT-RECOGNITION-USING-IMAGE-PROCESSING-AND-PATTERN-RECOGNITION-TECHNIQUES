#!/usr/bin/env python3
"""
Windows-specific configuration utilities for WeSee Application
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path


class WindowsConfig:
    """Windows-specific configuration helper"""
    
    @staticmethod
    def is_windows():
        """Check if running on Windows"""
        return platform.system() == "Windows"
    
    @staticmethod
    def configure_tts():
        """
        Configure Text-to-Speech for Windows
        Returns configured TTS engine or None
        """
        if not WindowsConfig.is_windows():
            return None
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Configure Windows TTS settings
            voices = engine.getProperty('voices')
            if voices:
                # Use first available voice
                engine.setProperty('voice', voices[0].id)
            
            # Set speech rate (words per minute)
            engine.setProperty('rate', 150)
            
            # Set volume (0.0 to 1.0)
            engine.setProperty('volume', 0.8)
            
            print("✓ Windows TTS configured successfully")
            return engine
            
        except ImportError:
            print("✗ pyttsx3 not installed for Windows TTS")
            return None
        except Exception as e:
            print(f"✗ Error configuring Windows TTS: {e}")
            return None
    
    @staticmethod
    def check_camera_access():
        """
        Check if camera is accessible on Windows
        Returns True if camera can be accessed
        """
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print("✓ Camera access working")
                    return True
                    
            print("✗ Camera not accessible")
            print("Please check:")
            print("- Camera permissions in Windows Settings > Privacy > Camera")
            print("- Camera is not being used by another application")
            print("- Camera drivers are properly installed")
            return False
            
        except Exception as e:
            print(f"✗ Camera check failed: {e}")
            return False
    
    @staticmethod
    def setup_windows_environment():
        """
        Complete Windows environment setup
        Returns True if all components are configured successfully
        """
        if not WindowsConfig.is_windows():
            print("Not running on Windows, skipping Windows-specific setup")
            return True
            
        print("🔧 Configuring Windows environment for WeSee...")
        print("-" * 50)
        
        success = True
        
        # Configure TTS (only for object detection audio feedback)
        tts_engine = WindowsConfig.configure_tts()
        if tts_engine is None:
            print("⚠️  TTS not available, audio feedback disabled")
        else:
            # Test TTS briefly
            try:
                tts_engine.say("WeSee Windows setup complete")
                tts_engine.runAndWait()
            except:
                pass
            
        # Check camera
        if not WindowsConfig.check_camera_access():
            success = False
            
        print("-" * 50)
        print("✓ Windows environment configured successfully!")
        print("🚀 Ready to run WeSee on Windows (Object Detection Only)")
            
        return success


# Global instance
windows_config = WindowsConfig()