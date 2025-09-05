#!/usr/bin/env python3
"""
Direct test of voice conditioning pipeline to identify where voice cloning fails.
"""

import sys
import os
import logging
import torch
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def debug_voice_conditioning():
    """Debug voice conditioning step by step."""
    print("🔍 Debugging Voice Conditioning Pipeline")
    print("=" * 60)
    
    try:
        from app.services.voice_service import VoiceService
        from app.utils.cuda_utils import cuda_manager
        
        # Create voice service instance
        print("📝 Initializing VoiceService...")
        voice_service = VoiceService()
        
        if not voice_service.is_model_loaded():
            print("❌ Model not loaded")
            return False
        
        print("✅ Model loaded successfully")
        
        # Get available voices
        voices = voice_service.get_voice_profiles()
        print(f"📂 Found {len(voices)} voices:")
        for voice in voices:
            print(f"  - {voice.name} (ID: {voice.id})")
            print(f"    Path: {voice.file_path}")
            print(f"    Exists: {os.path.exists(voice.file_path)}")
            
            # Check voice file properties
            if os.path.exists(voice.file_path):
                file_size = os.path.getsize(voice.file_path)
                print(f"    Size: {file_size} bytes ({file_size/1024:.1f} KB)")
        
        if not voices:
            print("❌ No voices available for testing")
            return False
        
        # Test with first available voice
        test_voice = voices[0]
        test_text = "This is a test to verify voice cloning works correctly."
        
        print(f"\n🎤 Testing voice conditioning with: {test_voice.name}")
        print(f"📝 Test text: {test_text}")
        print(f"📁 Voice file: {test_voice.file_path}")
        
        # Step 1: Test processor directly
        print("\n🔄 Step 1: Testing VibeVoice Processor directly...")
        try:
            formatted_text = f"Speaker 0: {test_text}"
            print(f"Formatted text: {formatted_text}")
            
            inputs = voice_service.processor(
                text=[formatted_text],
                voice_samples=[[test_voice.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            print(f"✅ Processor success. Input keys: {list(inputs.keys())}")
            
            # Analyze speech tensors specifically
            if "speech_tensors" in inputs:
                speech_tensors = inputs["speech_tensors"]
                print(f"  📊 Speech tensors shape: {speech_tensors.shape}")
                print(f"  📊 Speech tensors dtype: {speech_tensors.dtype}")
                print(f"  📊 Speech tensors device: {speech_tensors.device}")
                print(f"  📊 Speech tensors range: [{speech_tensors.min():.3f}, {speech_tensors.max():.3f}]")
                print(f"  📊 Speech tensors mean: {speech_tensors.mean():.3f}")
                print(f"  📊 Speech tensors std: {speech_tensors.std():.3f}")
                
                # Check if tensors are all zeros (indicates voice not processed)
                if torch.allclose(speech_tensors, torch.zeros_like(speech_tensors)):
                    print("  ⚠️  WARNING: Speech tensors are all zeros! Voice not processed.")
                else:
                    print("  ✅ Speech tensors contain data (voice processed)")
            else:
                print("  ❌ No speech_tensors in processor output!")
            
            if "speech_masks" in inputs:
                speech_masks = inputs["speech_masks"]
                print(f"  📊 Speech masks shape: {speech_masks.shape}")
                print(f"  📊 Speech masks sum: {speech_masks.sum()}")
            else:
                print("  ❌ No speech_masks in processor output!")
                
        except Exception as proc_error:
            print(f"❌ Processor failed: {proc_error}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 2: Test model generation with voice conditioning
        print("\n🔄 Step 2: Testing model generation with voice conditioning...")
        try:
            device = cuda_manager.device
            model_dtype = cuda_manager.dtype
            
            # Move inputs to device and ensure dtype consistency
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device, non_blocking=True)
                    if v.dtype != model_dtype and v.dtype.is_floating_point:
                        inputs[k] = inputs[k].to(model_dtype)
                        print(f"  🔄 Converted {k} from {v.dtype} to {model_dtype}")
            
            print("  🚀 Generating with voice conditioning...")
            with torch.amp.autocast('cuda', enabled=device.startswith("cuda"), dtype=model_dtype):
                outputs = voice_service.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.3,
                    tokenizer=voice_service.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=True,  # Enable verbose output to see what's happening
                    use_cache=True,
                )
            
            # Analyze generation outputs
            if hasattr(outputs, "speech_outputs") and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                print(f"  ✅ Generation success!")
                print(f"  📊 Output shape: {audio_tensor.shape}")
                print(f"  📊 Output dtype: {audio_tensor.dtype}")
                print(f"  📊 Output device: {audio_tensor.device}")
                
                # Convert to numpy for analysis
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)
                audio_array = audio_tensor.detach().cpu().numpy()
                
                print(f"  📊 Audio range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
                print(f"  📊 Audio mean: {audio_array.mean():.3f}")
                print(f"  📊 Audio std: {audio_array.std():.3f}")
                print(f"  📊 Duration: {len(audio_array) / 24000:.2f} seconds")
                
                # Check for conditioning effectiveness
                audio_variance = float(audio_array.var())
                print(f"  📊 Audio variance: {audio_variance:.6f}")
                
                if audio_variance > 0.001:
                    print("  ✅ Audio has good variance (likely real speech)")
                else:
                    print("  ⚠️  Low variance - may be placeholder or poor conditioning")
                
                return True
            else:
                print("  ❌ No speech output generated")
                return False
                
        except Exception as gen_error:
            print(f"❌ Generation failed: {gen_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_file_processing():
    """Test voice file processing independently."""
    print("\n🔧 Testing Voice File Processing")
    print("=" * 60)
    
    try:
        from app.services.voice_service import VoiceService
        import librosa
        
        voice_service = VoiceService()
        voices = voice_service.get_voice_profiles()
        
        if not voices:
            print("❌ No voices to test")
            return False
        
        test_voice = voices[0]
        print(f"🎤 Testing voice file: {test_voice.name}")
        print(f"📁 Path: {test_voice.file_path}")
        
        # Load audio directly with librosa
        try:
            audio_data, sr = librosa.load(test_voice.file_path, sr=24000)
            print(f"✅ Librosa load success:")
            print(f"  📊 Shape: {audio_data.shape}")
            print(f"  📊 Sample rate: {sr}")
            print(f"  📊 Duration: {len(audio_data) / sr:.2f} seconds")
            print(f"  📊 Range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            print(f"  📊 RMS: {np.sqrt(np.mean(audio_data**2)):.3f}")
            
            # Test if audio has actual content
            if np.std(audio_data) > 0.01:
                print("  ✅ Voice sample has good content")
            else:
                print("  ⚠️  Voice sample may be silent or low quality")
                
        except Exception as load_error:
            print(f"❌ Failed to load voice file: {load_error}")
            return False
        
        # Test processor with this voice
        print("\n🔄 Testing processor with voice file...")
        try:
            processor_inputs = voice_service.processor(
                text=["Speaker 0: Hello world"],
                voice_samples=[[test_voice.file_path]],
                padding=True,
                return_tensors="pt",
            )
            
            if "speech_tensors" in processor_inputs:
                speech_tensors = processor_inputs["speech_tensors"]
                print(f"✅ Voice processed into speech tensors:")
                print(f"  📊 Shape: {speech_tensors.shape}")
                print(f"  📊 Non-zero elements: {torch.count_nonzero(speech_tensors)}")
                print(f"  📊 Range: [{speech_tensors.min():.3f}, {speech_tensors.max():.3f}]")
                
                # Critical test: are speech tensors meaningful?
                if torch.count_nonzero(speech_tensors) == 0:
                    print("  ❌ CRITICAL: Speech tensors are all zeros!")
                    print("  🔍 This means voice is not being processed properly")
                    return False
                else:
                    print("  ✅ Speech tensors contain voice data")
                    return True
            else:
                print("❌ No speech_tensors generated by processor")
                return False
                
        except Exception as proc_error:
            print(f"❌ Processor test failed: {proc_error}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Voice file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive voice conditioning debug."""
    print("DJZ-VibeVoice Voice Conditioning Debug")
    print("=" * 60)
    
    # Test 1: Voice file processing
    voice_file_ok = test_voice_file_processing()
    
    # Test 2: Full voice conditioning pipeline
    voice_conditioning_ok = debug_voice_conditioning()
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC SUMMARY:")
    print(f"  Voice File Processing: {'✅ OK' if voice_file_ok else '❌ FAILED'}")
    print(f"  Voice Conditioning: {'✅ OK' if voice_conditioning_ok else '❌ FAILED'}")
    
    if not voice_file_ok:
        print("\n💡 RECOMMENDATION: Voice file processing is broken.")
        print("   Check voice file format, sample rate, and processor configuration.")
    elif not voice_conditioning_ok:
        print("\n💡 RECOMMENDATION: Voice conditioning in model generation is broken.")
        print("   The voice is processed correctly but not applied during generation.")
    else:
        print("\n🎉 Voice conditioning pipeline appears to be working!")
        print("   If voice still doesn't match, check model training/configuration.")
    
    return voice_file_ok and voice_conditioning_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
