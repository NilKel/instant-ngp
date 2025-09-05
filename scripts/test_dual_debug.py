#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/nilkel/Projects/instant-ngp')

import pyngp as ngp
import numpy as np

def test_dual_separate():
    print("🔍 Testing dual_separate mode step by step...")
    
    # Create testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.nerf.training.random_bg_color = False
    
    # Load a simple scene
    testbed.load_training_data("/home/nilkel/Projects/data/nerf_synthetic/drums/transforms_train.json")
    
    # Check the mode is set correctly
    print(f"🔍 Radiance head mode: {testbed.nerf.radiance_head_mode}")
    
    # Try to set dual_separate mode manually
    testbed.nerf.set_radiance_head_mode("dual_separate")
    print(f"🔍 After setting: {testbed.nerf.radiance_head_mode}")
    
    # Try a very small training step to see if it works
    testbed.training_step = 0
    print("🔍 Attempting minimal training step...")
    
    try:
        testbed.frame()
        print("✅ Frame succeeded")
    except Exception as e:
        print(f"❌ Frame failed: {e}")
        return False
    
    print("✅ Test completed successfully")
    return True

if __name__ == "__main__":
    test_dual_separate() 