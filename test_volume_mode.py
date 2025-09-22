#!/usr/bin/env python3

import json
import torch
import numpy as np

# Add the build directory to the path to import pyngp
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

import pyngp as ngp

def test_volume_mode():
    print("Testing Volume Mode Implementation")
    print("=" * 50)
    
    # Test configuration for volume mode
    config = {
        "pos_encoding": {
            "otype": "Frequency",
            "n_frequencies": 10
        },
        "dir_encoding": {
            "otype": "Frequency", 
            "n_frequencies": 4
        },
        "density_network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        },
        "rgb_network": {
            "otype": "FullyFusedMLP", 
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }
    }
    
    # Test different modes to compare
    modes = ["baseline", "surface", "volume"]
    
    for mode in modes:
        print(f"\n--- Testing {mode} mode ---")
        
        try:
            # Create network with specific mode
            network = ngp.NerfNetwork(
                n_pos_dims=3,
                n_dir_dims=3, 
                n_extra_dims=0,
                dir_offset=3,
                pos_encoding=config["pos_encoding"],
                dir_encoding=config["dir_encoding"],
                density_network=config["density_network"],
                rgb_network=config["rgb_network"],
                method=mode
            )
            
            print(f"✓ {mode} network created successfully")
            print(f"  - Input width: {network.input_width()}")
            print(f"  - Output width: {network.output_width()}")
            print(f"  - N params: {network.n_params()}")
            
            # Test inference
            batch_size = 1024
            input_data = torch.randn(network.input_width(), batch_size, dtype=torch.float32, device="cuda")
            output = torch.zeros(network.output_width(), batch_size, dtype=torch.float32, device="cuda")
            
            # Run inference
            network.inference(input_data, output)
            
            print(f"  - Inference successful: output shape {output.shape}")
            print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check for NaN/Inf values
            if torch.isnan(output).any():
                print(f"  ⚠ WARNING: NaN values detected in {mode} output")
            elif torch.isinf(output).any():
                print(f"  ⚠ WARNING: Inf values detected in {mode} output")
            else:
                print(f"  ✓ Output is finite and valid")
                
        except Exception as e:
            print(f"  ✗ Error in {mode} mode: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Volume mode test completed!")
    
    # Test that volume mode produces different outputs than surface mode
    if "volume" in modes and "surface" in modes:
        print("\nTesting that volume mode produces different outputs than surface mode...")
        
        try:
            # Create networks
            surface_net = ngp.NerfNetwork(
                n_pos_dims=3, n_dir_dims=3, n_extra_dims=0, dir_offset=3,
                pos_encoding=config["pos_encoding"], dir_encoding=config["dir_encoding"],
                density_network=config["density_network"], rgb_network=config["rgb_network"],
                method="surface"
            )
            
            volume_net = ngp.NerfNetwork(
                n_pos_dims=3, n_dir_dims=3, n_extra_dims=0, dir_offset=3,
                pos_encoding=config["pos_encoding"], dir_encoding=config["dir_encoding"],
                density_network=config["density_network"], rgb_network=config["rgb_network"],
                method="volume"
            )
            
            # Use same input
            input_data = torch.randn(surface_net.input_width(), 512, dtype=torch.float32, device="cuda")
            surface_output = torch.zeros(surface_net.output_width(), 512, dtype=torch.float32, device="cuda")
            volume_output = torch.zeros(volume_net.output_width(), 512, dtype=torch.float32, device="cuda")
            
            # Run inference
            surface_net.inference(input_data, surface_output)
            volume_net.inference(input_data, volume_output)
            
            # Compare outputs
            diff = torch.abs(surface_output - volume_output).mean()
            print(f"Mean absolute difference between surface and volume: {diff.item():.6f}")
            
            if diff.item() > 1e-6:
                print("✓ Volume mode produces meaningfully different outputs than surface mode")
            else:
                print("⚠ Volume and surface modes produce very similar outputs (may need different initialization)")
                
        except Exception as e:
            print(f"✗ Error comparing surface and volume modes: {e}")

if __name__ == "__main__":
    test_volume_mode() 