#!/usr/bin/env python3
"""
Test script để kiểm tra tính năng âm thanh
"""
import pyttsx3
import time

def test_audio():
    try:
        print("🔊 Testing Text-to-Speech...")
        engine = pyttsx3.init()
        engine.setProperty('rate', 235)
        engine.setProperty('volume', 1.0)
        
        # Test phát âm
        engine.say("Hello! Audio system is working")
        engine.runAndWait()
        
        engine.say("Person is 5 meters on LEFT")
        engine.runAndWait()
        
        engine.say("Car is 3 meters on FORWARD")
        engine.runAndWait()
        
        print("✅ Audio test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

if __name__ == "__main__":
    test_audio()
