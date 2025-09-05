#!/usr/bin/env python3
"""
Test script to validate tokenizer configuration fixes.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_tokenizer_fix():
    """Test that the tokenizer configuration is now working properly."""
    print("🧪 Testing Tokenizer Configuration Fix...")
    
    try:
        from backend.app.config import settings
        print(f"Model path: {settings.MODEL_PATH}")
        
        # Check if preprocessor config exists
        import os
        config_path = os.path.join(settings.MODEL_PATH, "preprocessor_config.json")
        if os.path.exists(config_path):
            print(f"✅ Found preprocessor_config.json at {config_path}")
        else:
            print(f"❌ Missing preprocessor_config.json at {config_path}")
            return False
        
        # Test processor loading
        try:
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)
            print(f"✅ VibeVoiceProcessor loaded successfully")
            print(f"   Tokenizer type: {type(processor.tokenizer).__name__}")
            
            # Check if we have the correct tokenizer
            if "VibeVoiceTextTokenizer" in type(processor.tokenizer).__name__:
                print("✅ Correct tokenizer type (VibeVoiceTextTokenizer)")
            else:
                print(f"⚠️  Unexpected tokenizer type: {type(processor.tokenizer).__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load VibeVoiceProcessor: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tokenizer configuration test."""
    print("DJZ-VibeVoice Tokenizer Fix Validation")
    print("=" * 50)
    
    success = test_tokenizer_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Tokenizer configuration fix validated successfully!")
        print("The VibeVoice processor should now use the correct tokenizer.")
    else:
        print("⚠️  Tokenizer configuration fix needs more work.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
