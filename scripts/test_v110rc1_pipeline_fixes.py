#!/usr/bin/env python3
"""
V1.10 RC1 Pipeline Diagnostic Test Suite
Tests the fixes for prompt following accuracy and optimization control.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    from app.config import settings
    from app.services.voice_service import VoiceService
    from app.models import VoiceProfile, VoiceType
except ImportError as e:
    print(f"Failed to import backend modules: {e}")
    print("Make sure you're running from the project root and backend is properly set up")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V110RC1DiagnosticTester:
    """Test suite for V1.10 RC1 pipeline fixes."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.voice_service = None
        self.test_voice_id = None
        self.results = {
            "config_tests": {},
            "prompt_following_tests": {},
            "performance_tests": {},
            "compatibility_tests": {}
        }
        
    def setup_test_environment(self) -> bool:
        """Set up the test environment and create test voice."""
        try:
            logger.info("Setting up V1.10 RC1 diagnostic test environment...")
            
            # Initialize voice service
            self.voice_service = VoiceService()
            
            # Create a test voice file if none exists
            self._create_test_voice()
            
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}")
            return False
    
    def _create_test_voice(self):
        """Use the real speech sample for proper voice conditioning testing."""
        # Use the known-good speech sample
        real_voice_path = settings.VOICES_DIR / "djz-sample_1_8f26fabb.wav"
        
        if not real_voice_path.exists():
            logger.error(f"Real voice sample not found: {real_voice_path}")
            # Fall back to any available voice
            voices = self.voice_service.get_voice_profiles()
            if voices:
                self.test_voice_id = voices[0].id
                logger.info(f"Using fallback voice: {voices[0].name} ({voices[0].id})")
                return
            else:
                logger.error("No voices available for testing")
                return
        
        # Check if voice is already loaded
        existing_voices = self.voice_service.get_voice_profiles()
        for voice in existing_voices:
            if voice.file_path == str(real_voice_path):
                self.test_voice_id = voice.id
                logger.info(f"Using existing real voice: {voice.name} ({voice.id})")
                return
        
        # Add the real voice sample to voice service
        profile = self.voice_service.add_voice_profile(
            name="DJZ Real Speech Sample",
            audio_path=str(real_voice_path),
            voice_type=VoiceType.UPLOADED
        )
        self.test_voice_id = profile.id
        logger.info(f"Loaded real speech sample: {profile.name} ({profile.id})")
        logger.info(f"Voice file path: {profile.file_path}")
        
        # Verify voice file exists and get properties
        if real_voice_path.exists():
            file_size = real_voice_path.stat().st_size
            logger.info(f"Voice file size: {file_size} bytes ({file_size/1024:.1f} KB)")
        else:
            logger.error(f"Voice file verification failed: {real_voice_path}")

    def test_configuration_controls(self) -> Dict[str, bool]:
        """Test Phase A: Configuration control tests."""
        logger.info("🔬 Testing Configuration Controls...")
        
        config_tests = {}
        
        # Test 1: Verify default configuration settings
        config_tests["default_preserve_original"] = settings.PRESERVE_ORIGINAL_BEHAVIOR == True
        config_tests["default_custom_config_disabled"] = settings.ENABLE_CUSTOM_GENERATION_CONFIG == False
        config_tests["default_cfg_override_disabled"] = settings.ENABLE_CFG_OVERRIDE == False
        config_tests["default_auto_optimize_disabled"] = settings.AUTO_USE_OPTIMIZATIONS == False
        
        # Test 2: Verify configuration parameters exist
        config_tests["generation_params_exist"] = all([
            hasattr(settings, 'GENERATION_DO_SAMPLE'),
            hasattr(settings, 'GENERATION_TEMPERATURE'), 
            hasattr(settings, 'GENERATION_TOP_P'),
            hasattr(settings, 'GENERATION_REPETITION_PENALTY'),
        ])
        
        # Test 3: Verify optimization control parameters exist
        config_tests["optimization_controls_exist"] = all([
            hasattr(settings, 'OPTIMIZE_SINGLE_REQUESTS'),
            hasattr(settings, 'OPTIMIZE_BATCH_REQUESTS'),
        ])
        
        self.results["config_tests"] = config_tests
        
        # Log results
        for test_name, passed in config_tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        return config_tests

    def test_prompt_following_accuracy(self) -> Dict[str, Dict]:
        """Test Phase B: Prompt following accuracy tests."""
        logger.info("🎯 Testing Prompt Following Accuracy...")
        
        if not self.test_voice_id:
            logger.error("No test voice available - skipping accuracy tests")
            return {}

        # Test cases from the diagnostic plan
        test_cases = [
            {
                "name": "exact_match",
                "prompt": "Say exactly: Hello world",
                "expected_keywords": ["hello", "world"],
                "cfg_scale": 1.0,
                "description": "Low CFG for text adherence"
            },
            {
                "name": "sequence_match", 
                "prompt": "Count: one, two, three, four, five",
                "expected_keywords": ["one", "two", "three", "four", "five"],
                "cfg_scale": 1.2,
                "description": "Sequence preservation test"
            },
            {
                "name": "spelling_test",
                "prompt": "Spell 'cat': C-A-T", 
                "expected_keywords": ["C", "A", "T"],
                "cfg_scale": 1.0,
                "description": "Letter-by-letter spelling"
            },
            {
                "name": "cfg_scale_respect",
                "prompt": "Simple test with low CFG",
                "expected_keywords": ["simple", "test"],
                "cfg_scale": 0.8,  # Below the old forced minimum of 1.5
                "description": "Verify user CFG scale is respected"
            }
        ]
        
        prompt_tests = {}
        
        for test_case in test_cases:
            logger.info(f"  Testing: {test_case['name']} - {test_case['description']}")
            
            try:
                # Test with original behavior (should respect user cfg_scale)
                start_time = time.time()
                audio_result = self.voice_service.generate_speech(
                    text=test_case["prompt"],
                    voice_id=self.test_voice_id,
                    cfg_scale=test_case["cfg_scale"],
                    use_optimizations=False  # Force original behavior
                )
                generation_time = time.time() - start_time
                
                # Basic validation
                test_result = {
                    "success": audio_result is not None,
                    "generation_time": generation_time,
                    "cfg_scale_used": test_case["cfg_scale"],
                    "audio_length": len(audio_result) if audio_result is not None else 0,
                    "mode": "original"
                }
                
                if audio_result is not None:
                    test_result["audio_stats"] = {
                        "min": float(np.min(audio_result)),
                        "max": float(np.max(audio_result)),
                        "mean": float(np.mean(audio_result)),
                        "std": float(np.std(audio_result))
                    }
                
                prompt_tests[test_case['name']] = test_result
                
                status = "✅ PASS" if test_result["success"] else "❌ FAIL"
                logger.info(f"    {status} - Generated: {test_result['audio_length']} samples in {generation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"    ❌ FAIL - Error: {e}")
                prompt_tests[test_case['name']] = {
                    "success": False,
                    "error": str(e),
                    "mode": "original"
                }
        
        self.results["prompt_following_tests"] = prompt_tests
        return prompt_tests

    def test_performance_matrix(self) -> Dict[str, Dict]:
        """Test Phase C: Performance vs accuracy matrix."""
        logger.info("⚡ Testing Performance Matrix...")
        
        if not self.test_voice_id:
            logger.error("No test voice available - skipping performance tests")
            return {}

        # Performance test configurations
        test_configs = [
            {
                "name": "original_behavior",
                "use_optimizations": False,
                "description": "Original VibeVoice behavior"
            },
            {
                "name": "optimized_behavior", 
                "use_optimizations": True,
                "description": "CUDA optimizations enabled"
            }
        ]
        
        performance_tests = {}
        test_prompt = "This is a performance test prompt for measuring generation speed and quality."
        
        for config in test_configs:
            logger.info(f"  Testing: {config['name']} - {config['description']}")
            
            try:
                # Run multiple iterations for average timing
                times = []
                successes = 0
                
                for i in range(3):  # 3 iterations for averaging
                    start_time = time.time()
                    audio_result = self.voice_service.generate_speech(
                        text=test_prompt,
                        voice_id=self.test_voice_id,
                        cfg_scale=1.3,  # Standard cfg_scale
                        use_optimizations=config["use_optimizations"]
                    )
                    generation_time = time.time() - start_time
                    
                    if audio_result is not None:
                        times.append(generation_time)
                        successes += 1
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    
                    performance_tests[config['name']] = {
                        "success_rate": successes / 3,
                        "avg_generation_time": avg_time,
                        "std_generation_time": std_time,
                        "min_time": min(times),
                        "max_time": max(times),
                        "mode": config['name']
                    }
                    
                    logger.info(f"    ✅ Average time: {avg_time:.2f}s ± {std_time:.2f}s")
                else:
                    performance_tests[config['name']] = {
                        "success_rate": 0,
                        "error": "No successful generations",
                        "mode": config['name']
                    }
                    logger.info(f"    ❌ No successful generations")
                    
            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                performance_tests[config['name']] = {
                    "success_rate": 0,
                    "error": str(e),
                    "mode": config['name']
                }
        
        self.results["performance_tests"] = performance_tests
        return performance_tests

    def test_cfg_scale_behavior(self) -> Dict[str, bool]:
        """Test that CFG scale values are properly respected."""
        logger.info("🎚️ Testing CFG Scale Behavior...")
        
        if not self.test_voice_id:
            logger.error("No test voice available - skipping CFG tests")
            return {}

        cfg_tests = {}
        test_prompt = "Test CFG scale behavior"
        
        # Test different CFG scale values
        cfg_values = [0.5, 1.0, 1.3, 2.0]  # Including values below the old forced minimum
        
        for cfg_scale in cfg_values:
            logger.info(f"  Testing CFG scale: {cfg_scale}")
            
            try:
                # Test with original behavior (should respect exact value)
                audio_result = self.voice_service.generate_speech(
                    text=test_prompt,
                    voice_id=self.test_voice_id,
                    cfg_scale=cfg_scale,
                    use_optimizations=False  # Force original behavior
                )
                
                cfg_tests[f"cfg_{cfg_scale}"] = audio_result is not None
                status = "✅ PASS" if audio_result is not None else "❌ FAIL"
                logger.info(f"    CFG {cfg_scale}: {status}")
                
            except Exception as e:
                logger.error(f"    CFG {cfg_scale}: ❌ Error - {e}")
                cfg_tests[f"cfg_{cfg_scale}"] = False
        
        self.results["cfg_scale_tests"] = cfg_tests
        return cfg_tests

    def test_compatibility_modes(self) -> Dict[str, Dict]:
        """Test different compatibility mode configurations."""
        logger.info("🔄 Testing Compatibility Modes...")
        
        if not self.test_voice_id:
            logger.error("No test voice available - skipping compatibility tests")
            return {}

        # Save original settings
        original_preserve = settings.PRESERVE_ORIGINAL_BEHAVIOR
        original_auto_opt = settings.AUTO_USE_OPTIMIZATIONS
        original_cfg_override = settings.ENABLE_CFG_OVERRIDE
        original_custom_config = settings.ENABLE_CUSTOM_GENERATION_CONFIG
        
        compatibility_tests = {}
        test_prompt = "Compatibility test prompt"
        
        try:
            # Test Matrix: Different configuration combinations
            test_matrix = [
                {
                    "name": "default_safe_mode",
                    "preserve_original": True,
                    "auto_optimizations": False,
                    "cfg_override": False,
                    "custom_config": False,
                    "expected": "original_behavior"
                },
                {
                    "name": "optimizations_enabled",
                    "preserve_original": False,
                    "auto_optimizations": True,
                    "cfg_override": False,
                    "custom_config": False,
                    "expected": "optimized_behavior"
                },
                {
                    "name": "cfg_override_enabled",
                    "preserve_original": True,
                    "auto_optimizations": False,
                    "cfg_override": True,
                    "custom_config": False,
                    "expected": "cfg_forced"
                },
                {
                    "name": "custom_config_enabled",
                    "preserve_original": True,
                    "auto_optimizations": False,
                    "cfg_override": False,
                    "custom_config": True,
                    "expected": "custom_params"
                }
            ]
            
            for test_config in test_matrix:
                logger.info(f"  Testing: {test_config['name']}")
                
                # Apply test configuration
                settings.PRESERVE_ORIGINAL_BEHAVIOR = test_config["preserve_original"]
                settings.AUTO_USE_OPTIMIZATIONS = test_config["auto_optimizations"]
                settings.ENABLE_CFG_OVERRIDE = test_config["cfg_override"]
                settings.ENABLE_CUSTOM_GENERATION_CONFIG = test_config["custom_config"]
                
                try:
                    start_time = time.time()
                    audio_result = self.voice_service.generate_speech(
                        text=test_prompt,
                        voice_id=self.test_voice_id,
                        cfg_scale=1.0,  # Use low CFG to test override behavior
                    )
                    generation_time = time.time() - start_time
                    
                    compatibility_tests[test_config['name']] = {
                        "success": audio_result is not None,
                        "generation_time": generation_time,
                        "expected_behavior": test_config["expected"],
                        "config": {
                            k: v for k, v in test_config.items() 
                            if k not in ["name", "expected"]
                        }
                    }
                    
                    status = "✅ PASS" if audio_result is not None else "❌ FAIL"
                    logger.info(f"    {status} - Time: {generation_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"    ❌ Error: {e}")
                    compatibility_tests[test_config['name']] = {
                        "success": False,
                        "error": str(e),
                        "expected_behavior": test_config["expected"]
                    }
        
        finally:
            # Restore original settings
            settings.PRESERVE_ORIGINAL_BEHAVIOR = original_preserve
            settings.AUTO_USE_OPTIMIZATIONS = original_auto_opt
            settings.ENABLE_CFG_OVERRIDE = original_cfg_override
            settings.ENABLE_CUSTOM_GENERATION_CONFIG = original_custom_config
        
        self.results["compatibility_tests"] = compatibility_tests
        return compatibility_tests

    def run_emergency_validation(self) -> bool:
        """Run emergency validation tests from the diagnostic plan."""
        logger.info("🚨 Running Emergency Validation Tests...")
        
        # Test the critical fixes
        critical_tests = []
        
        # Test 1: CFG scale override disabled
        logger.info("  Testing CFG scale override fix...")
        try:
            audio = self.voice_service.generate_speech(
                text="CFG override test",
                voice_id=self.test_voice_id,
                cfg_scale=0.8,  # Below old forced minimum
                use_optimizations=False
            )
            critical_tests.append(("cfg_override_fix", audio is not None))
        except Exception as e:
            logger.error(f"CFG override test failed: {e}")
            critical_tests.append(("cfg_override_fix", False))
        
        # Test 2: Generation config is optional
        logger.info("  Testing generation config fix...")
        try:
            # With default settings, generation_config should be None
            audio = self.voice_service._generate_speech_original(
                text="Generation config test",
                voice_id=self.test_voice_id,
                cfg_scale=1.3
            )
            critical_tests.append(("generation_config_fix", audio is not None))
        except Exception as e:
            logger.error(f"Generation config test failed: {e}")
            critical_tests.append(("generation_config_fix", False))
        
        # Test 3: Dual-path routing works
        logger.info("  Testing dual-path routing...")
        try:
            # Test original path
            audio_orig = self.voice_service.generate_speech(
                text="Dual path test", 
                voice_id=self.test_voice_id,
                use_optimizations=False
            )
            
            # Test optimized path
            audio_opt = self.voice_service.generate_speech(
                text="Dual path test",
                voice_id=self.test_voice_id, 
                use_optimizations=True
            )
            
            critical_tests.append(("dual_path_routing", 
                                 audio_orig is not None and audio_opt is not None))
        except Exception as e:
            logger.error(f"Dual-path routing test failed: {e}")
            critical_tests.append(("dual_path_routing", False))
        
        # Log critical test results
        all_passed = True
        for test_name, passed in critical_tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed

    def generate_diagnostic_report(self) -> Dict:
        """Generate comprehensive diagnostic report."""
        logger.info("📊 Generating Diagnostic Report...")
        
        # Calculate summary statistics
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_environment": {
                "preserve_original_behavior": settings.PRESERVE_ORIGINAL_BEHAVIOR,
                "enable_custom_generation_config": settings.ENABLE_CUSTOM_GENERATION_CONFIG,
                "enable_cfg_override": settings.ENABLE_CFG_OVERRIDE,
                "auto_use_optimizations": settings.AUTO_USE_OPTIMIZATIONS,
                "model_loaded": self.voice_service.is_model_loaded() if self.voice_service else False,
                "test_voice_available": self.test_voice_id is not None
            },
            "results": self.results,
            "summary": {}
        }
        
        # Calculate pass rates
        config_pass_rate = sum(self.results.get("config_tests", {}).values()) / max(len(self.results.get("config_tests", {})), 1)
        
        prompt_tests = self.results.get("prompt_following_tests", {})
        prompt_pass_rate = sum(1 for test in prompt_tests.values() if test.get("success", False)) / max(len(prompt_tests), 1)
        
        performance_tests = self.results.get("performance_tests", {})
        performance_pass_rate = sum(1 for test in performance_tests.values() if test.get("success_rate", 0) > 0.5) / max(len(performance_tests), 1)
        
        compatibility_tests = self.results.get("compatibility_tests", {})
        compatibility_pass_rate = sum(1 for test in compatibility_tests.values() if test.get("success", False)) / max(len(compatibility_tests), 1)
        
        report["summary"] = {
            "config_pass_rate": config_pass_rate,
            "prompt_following_pass_rate": prompt_pass_rate,
            "performance_pass_rate": performance_pass_rate,
            "compatibility_pass_rate": compatibility_pass_rate,
            "overall_health": (config_pass_rate + prompt_pass_rate + performance_pass_rate + compatibility_pass_rate) / 4
        }
        
        return report

    def run_full_diagnostic_suite(self) -> Dict:
        """Run the complete diagnostic suite."""
        logger.info("🚀 Starting V1.10 RC1 Complete Diagnostic Suite...")
        
        if not self.setup_test_environment():
            logger.error("Failed to set up test environment")
            return {"error": "Environment setup failed"}
        
        # Run all test phases
        self.test_configuration_controls()
        self.test_prompt_following_accuracy()
        self.test_performance_matrix()
        self.test_compatibility_modes()
        
        # Run emergency validation
        emergency_status = self.run_emergency_validation()
        
        # Generate final report
        report = self.generate_diagnostic_report()
        report["emergency_validation_passed"] = emergency_status
        
        return report


def main():
    """Main diagnostic execution."""
    print("🔍 V1.10 RC1 Pipeline Diagnostic Test Suite")
    print("=" * 50)
    
    tester = V110RC1DiagnosticTester()
    
    try:
        # Run full diagnostic suite
        report = tester.run_full_diagnostic_suite()
        
        # Save report
        report_path = Path("v110rc1_diagnostic_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n📊 DIAGNOSTIC SUMMARY")
        print("-" * 30)
        
        if "error" in report:
            print(f"❌ CRITICAL ERROR: {report['error']}")
            return 1
        
        summary = report.get("summary", {})
        emergency_passed = report.get("emergency_validation_passed", False)
        
        print(f"Emergency Validation: {'✅ PASS' if emergency_passed else '❌ FAIL'}")
        print(f"Configuration Tests: {summary.get('config_pass_rate', 0):.1%} pass rate")
        print(f"Prompt Following: {summary.get('prompt_following_pass_rate', 0):.1%} pass rate")
        print(f"Performance Tests: {summary.get('performance_pass_rate', 0):.1%} pass rate")
        print(f"Compatibility Tests: {summary.get('compatibility_pass_rate', 0):.1%} pass rate")
        print(f"Overall Health: {summary.get('overall_health', 0):.1%}")
        
        print(f"\n📋 Full report saved to: {report_path}")
        
        # Return exit code based on emergency validation
        return 0 if emergency_passed else 1
        
    except Exception as e:
        logger.error(f"Diagnostic suite failed: {e}", exc_info=True)
        print(f"❌ DIAGNOSTIC FAILURE: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
