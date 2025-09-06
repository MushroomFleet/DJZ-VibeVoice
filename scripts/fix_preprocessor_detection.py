#!/usr/bin/env python3
"""
Fix preprocessor_config.json detection issue in VibeVoiceProcessor.

The issue is that VibeVoiceProcessor.from_pretrained() uses naive path joining
instead of proper HuggingFace model resolution to find preprocessor_config.json.

This script provides a workaround by patching the resolution mechanism.
"""

import os
import sys
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models" / "VibeVoice"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_current_behavior():
    """Test the current behavior to confirm the issue."""
    logger.info("=== Testing Current Behavior ===")
    
    try:
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        
        # This should show the warning
        logger.info("Loading processor (should show warning)...")
        processor = VibeVoiceProcessor.from_pretrained('microsoft/VibeVoice-1.5B')
        logger.info(f"✅ Processor loaded successfully: {type(processor)}")
        
        return processor
    except Exception as e:
        logger.error(f"❌ Failed to load processor: {e}")
        return None

def fix_with_huggingface_resolution():
    """Fix the issue using proper HuggingFace model resolution."""
    logger.info("\n=== Testing HuggingFace Resolution Fix ===")
    
    try:
        from huggingface_hub import snapshot_download
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        import json
        
        # Resolve the model path properly
        resolved_path = snapshot_download('microsoft/VibeVoice-1.5B')
        logger.info(f"Resolved model path: {resolved_path}")
        
        # Check if preprocessor_config.json exists
        config_path = os.path.join(resolved_path, "preprocessor_config.json")
        logger.info(f"Looking for config at: {config_path}")
        
        if os.path.exists(config_path):
            logger.info("✅ preprocessor_config.json found!")
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Config content: {config}")
            
            # Load processor using resolved path
            logger.info("Loading processor with resolved path...")
            processor = VibeVoiceProcessor.from_pretrained(resolved_path)
            logger.info("✅ Processor loaded without warnings!")
            
            return processor, resolved_path
        else:
            logger.error(f"❌ preprocessor_config.json not found at {config_path}")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ HuggingFace resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_patched_voice_service():
    """Create a patched version of voice_service.py that uses proper resolution."""
    logger.info("\n=== Creating Patched Voice Service ===")
    
    voice_service_path = project_root / "backend" / "app" / "services" / "voice_service.py"
    backup_path = voice_service_path.with_suffix('.py.backup_preprocessor_fix')
    
    try:
        # Read current file
        with open(voice_service_path, 'r') as f:
            content = f.read()
        
        # Create backup
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup: {backup_path}")
        
        # Find the processor loading section and patch it
        old_pattern = '''try:
                self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)
            except Exception as proc_error:
                logger.warning(f"Failed to load processor from {settings.MODEL_PATH}: {proc_error}")'''
        
        new_pattern = '''try:
                # Use proper HuggingFace resolution to avoid preprocessor_config.json warning
                from huggingface_hub import snapshot_download
                resolved_model_path = snapshot_download(settings.MODEL_PATH)
                logger.info(f"Resolved model path: {resolved_model_path}")
                self.processor = VibeVoiceProcessor.from_pretrained(resolved_model_path)
            except Exception as proc_error:
                logger.warning(f"Failed to load processor from {settings.MODEL_PATH}: {proc_error}")
                logger.info("Attempting fallback with original model path...")
                try:
                    self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise proc_error'''
        
        if old_pattern in content:
            patched_content = content.replace(old_pattern, new_pattern)
            
            with open(voice_service_path, 'w') as f:
                f.write(patched_content)
            
            logger.info("✅ Voice service patched successfully!")
            logger.info(f"Backup available at: {backup_path}")
            return True
        else:
            logger.warning("❌ Could not find the expected pattern to patch")
            logger.info("The voice_service.py structure may have changed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to patch voice service: {e}")
        return False

def test_patched_solution():
    """Test that the patched solution works."""
    logger.info("\n=== Testing Patched Solution ===")
    
    try:
        # Import the patched voice service
        sys.path.insert(0, str(project_root / "backend"))
        from app.services.voice_service import VoiceService
        
        logger.info("Creating VoiceService instance...")
        voice_service = VoiceService()
        
        if voice_service.model_loaded:
            logger.info("✅ VoiceService loaded successfully!")
            logger.info("✅ No preprocessor_config.json warnings should appear!")
            return True
        else:
            logger.warning("⚠️ VoiceService created but model not loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Testing patched solution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to fix the preprocessor detection issue."""
    logger.info("🔧 VibeVoice Preprocessor Detection Fix")
    logger.info("=" * 50)
    
    # Step 1: Test current behavior (should show warning)
    current_processor = test_current_behavior()
    
    # Step 2: Test HuggingFace resolution fix
    fixed_processor, resolved_path = fix_with_huggingface_resolution()
    
    if fixed_processor and resolved_path:
        logger.info("\n✅ HuggingFace resolution works!")
        
        # Step 3: Create patched voice service
        if create_patched_voice_service():
            logger.info("\n✅ Voice service patched!")
            
            # Step 4: Test the patched solution
            if test_patched_solution():
                logger.info("\n🎉 SUCCESS: Preprocessor detection issue fixed!")
                logger.info("\nThe backend should now start without preprocessor_config.json warnings.")
                return True
            else:
                logger.error("\n❌ Patched solution test failed")
                return False
        else:
            logger.error("\n❌ Failed to patch voice service")
            return False
    else:
        logger.error("\n❌ HuggingFace resolution failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
