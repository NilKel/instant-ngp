#!/usr/bin/env python3

import pyngp as ngp
import os

# Test surface loss with different background colors
def test_surface_loss_with_background():
    testbed = ngp.Testbed()
    
    # Load the fox scene
    testbed.load_training_data("data/nerf/fox")
    
    # Load surface configuration
    testbed.reload_network_from_file("configs/nerf/surface.json")
    
    # Test with black background (default)
    print("Testing with black background...")
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    
    # Run a few training steps
    for i in range(10):
        testbed.training_step()
        if i % 5 == 0:
            print(f"Step {i}: Loss = {testbed.nerf_loss}")
    
    # Test with white background
    print("\nTesting with white background...")
    testbed.background_color = [1.0, 1.0, 1.0, 1.0]
    
    # Run a few more training steps
    for i in range(10):
        testbed.training_step()
        if i % 5 == 0:
            print(f"Step {i+10}: Loss = {testbed.nerf_loss}")
    
    print("\nSurface loss test completed successfully!")
    print("The surface loss should automatically include the background color as a separate component.")

if __name__ == "__main__":
    test_surface_loss_with_background() 