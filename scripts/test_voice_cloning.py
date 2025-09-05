#!/usr/bin/env python3
"""
Test script to validate voice cloning functionality.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_voice_cloning():
    """Test voice cloning with detailed diagnostics."""
    print("🎯 Testing Voice Cloning Functionality")
    print("=" * 50)
    
    try:
        from backend.app.services.voice_service import VoiceService
        
        # Create voice service instance
        print("📝 Initializing VoiceService...")
        voice_service = VoiceService()
        
        if not voice_service.is_model_loaded():
            print("❌ Model not loaded, cannot test voice cloning")
            return False
        
        print("✅ Model loaded successfully")
        
        # Get available voices
        voices = voice_service.get_voice_profiles()
        print(f"📂 Found {len(voices)} voices:")
        for voice in voices:
            print(f"  - {voice.name} (ID: {voice.id[:8]}...)")
            print(f"    Path: {voice.file_path}")
            print(f"    Exists: {os.path.exists(voice.file_path)}")
        
        if not voices:
            print("❌ No voices available for testing")
            return False
        
        # Test with first available voice
        test_voice = voices[0]
        test_text = "Hello, this is a test of voice cloning functionality."
        
        print(f"\n🎤 Testing voice cloning with: {test_voice.name}")
        print(f"📝 Test text: {test_text}")
        print(f"📁 Voice file: {test_voice.file_path}")
        
        # Generate speech
        print("\n🔄 Starting voice generation...")
        audio_array = voice_service.generate_speech(
            text=test_text,
            voice_id=test_voice.id,
            num_speakers=1,
            cfg_scale=1.3
        )
        
        if audio_array is not None:
            print(f"✅ Voice generation successful!")
            print(f"   Audio shape: {audio_array.shape}")
            print(f"   Audio dtype: {audio_array.dtype}")
            print(f"   Audio range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
            print(f"   Duration: {len(audio_array) / 24000:.2f} seconds")
            
            # Check if it's not just the placeholder audio
            if len(audio_array) > 0:
                # Simple check - placeholder audio has very regular patterns
                audio_variance = float(audio_array.var())
                print(f"   Audio variance: {audio_variance:.6f}")
                
                if audio_variance > 0.001:  # Real speech should have more variance
                    print("✅ Audio appears to be real speech (good variance)")
                    return True
                else:
                    print("⚠️  Audio may be placeholder (low variance)")
                    return False
            else:
                print("❌ Empty audio array")
                return False
        else:
            print("❌ Voice generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run voice cloning test."""
    print("DJZ-VibeVoice Voice Cloning Test")
    print("=" * 50)
    
    success = test_voice_cloning()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Voice cloning test completed successfully!")
        print("Check the logs above for detailed voice processing information.")
    else:
        print("⚠️  Voice cloning test failed or issues detected.")
        print("Review the diagnostic logs to identify the problem.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
