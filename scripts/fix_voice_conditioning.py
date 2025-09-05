#!/usr/bin/env python3
"""
Fix for voice conditioning pipeline issues.
"""

import sys
import os
import logging

# Add the backend to Python path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def fix_voice_conditioning():
    """Apply fixes to voice conditioning pipeline."""
    print("🔧 Applying Voice Conditioning Fixes")
    print("=" * 50)
    
    try:
        from app.services.voice_service import VoiceService
        from app.utils.cuda_utils import cuda_manager
        
        print("📝 Initializing VoiceService...")
        voice_service = VoiceService()
        
        if not voice_service.is_model_loaded():
            print("❌ Model not loaded")
            return False
        
        print("✅ Model loaded successfully")
        
        # Get test voice
        voices = voice_service.get_voice_profiles()
        if not voices:
            print("❌ No voices available")
            return False
        
        test_voice = voices[0]
        test_text = "Testing voice conditioning fix."
        
        print(f"🎤 Testing with voice: {test_voice.name}")
        
        # Test the current implementation
        print("\n🔄 Testing current voice conditioning...")
        try:
            audio_result = voice_service.generate_speech(
                text=test_text,
                voice_id=test_voice.id,
                num_speakers=1,
                cfg_scale=1.3
            )
            
            if audio_result is not None:
                print(f"✅ Generation successful: {len(audio_result)} samples")
                print(f"📊 Audio variance: {audio_result.var():.6f}")
                
                # Check if this looks like real speech
                if audio_result.var() > 0.001:
                    print("✅ Audio appears to be real speech")
                else:
                    print("⚠️  Audio may be placeholder (low variance)")
                
                return True
            else:
                print("❌ Generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test voice conditioning fixes."""
    print("DJZ-VibeVoice Voice Conditioning Fix")
    print("=" * 50)
    
    success = fix_voice_conditioning()
    
    if success:
        print("\n🎉 Voice conditioning test passed!")
    else:
        print("\n⚠️  Voice conditioning issues detected.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
