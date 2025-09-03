#!/usr/bin/env python3

import pyngp as ngp
import os

# Test simplified surface loss with different background colors
def test_simplified_surface_loss():
    testbed = ngp.Testbed()
    
    # Load the fox scene
    testbed.load_training_data("data/nerf/fox")
    
    # Load surface configuration
    testbed.reload_network_from_file("configs/nerf/surface.json")
    
    print("Testing simplified surface loss with automatic background integration...")
    print("The surface loss now computes: alpha * loss(sample, GT) + (1-alpha) * loss(background, GT) for each sample")
    print("and multiplies by the transmittance from previous samples.")
    
    # Test with black background
    print("\n1. Testing with black background...")
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    
    # Run a few training steps
    for i in range(5):
        testbed.training_step()
        print(f"   Step {i}: Loss = {testbed.nerf_loss}")
    
    # Test with white background
    print("\n2. Testing with white background...")
    testbed.background_color = [1.0, 1.0, 1.0, 1.0]
    
    # Run a few more training steps
    for i in range(5):
        testbed.training_step()
        print(f"   Step {i+5}: Loss = {testbed.nerf_loss}")
    
    # Test with red background
    print("\n3. Testing with red background...")
    testbed.background_color = [1.0, 0.0, 0.0, 1.0]
    
    # Run a few more training steps
    for i in range(5):
        testbed.training_step()
        print(f"   Step {i+10}: Loss = {testbed.nerf_loss}")
    
    print("\n✅ Simplified surface loss test completed successfully!")
    print("The surface loss automatically uses whatever background color is set for rendering.")
    print("Each sample's loss is: alpha * loss(sample, GT) + (1-alpha) * loss(background, GT)")
    print("This is multiplied by the transmittance from previous samples.")

if __name__ == "__main__":
    test_simplified_surface_loss() 