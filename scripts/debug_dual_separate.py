#!/usr/bin/env python3

import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import testbed
sys.path.append(str(Path(__file__).parent.parent))

import pyngp as ngp

def main():
    parser = argparse.ArgumentParser(description="Debug dual_separate mode rendering")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene JSON file")
    parser.add_argument("--snapshot", type=str, help="Path to trained snapshot")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512], help="Render resolution")
    parser.add_argument("--output_dir", type=str, default="debug_output", help="Output directory for debug images")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 [DEBUG] Initializing testbed with dual_separate configuration...")
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    
    # Load scene
    print(f"🔍 [DEBUG] Loading scene: {args.scene}")
    testbed.load_training_data(args.scene)
    
    # Load snapshot if provided
    if args.snapshot:
        print(f"🔍 [DEBUG] Loading snapshot: {args.snapshot}")
        testbed.load_snapshot(args.snapshot)
    else:
        print("🔍 [DEBUG] No snapshot provided, using random weights")
    
    # Set dual_separate configuration
    print("🔍 [DEBUG] Configuring network for dual_separate mode...")
    
    # First check if we can set the radiance head mode
    try:
        testbed.nerf.radiance_head_mode = "dual_separate"
        print(f"✅ Set radiance head mode to: {testbed.nerf.radiance_head_mode}")
    except Exception as e:
        print(f"❌ Failed to set radiance head mode: {e}")
        return
    
    # Test basic rendering first
    print("🔍 [DEBUG] Testing basic rendering...")
    try:
        width, height = args.resolution
        image = testbed.render(width, height, 1, True)
        print(f"✅ Basic rendering successful: {image.shape}")
        
        # Save basic render
        from PIL import Image
        img_array = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_array).save(f"{args.output_dir}/basic_render.png")
        print(f"💾 Saved basic render to {args.output_dir}/basic_render.png")
        
    except Exception as e:
        print(f"❌ Basic rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test dual separate rendering if available
    print("🔍 [DEBUG] Testing dual separate rendering...")
    try:
        # Check if render_dual_separate method exists
        if hasattr(testbed, 'render_dual_separate'):
            surface_img, volume_img = testbed.render_dual_separate(width, height, 1, True)
            print(f"✅ Dual separate rendering successful")
            print(f"   Surface image shape: {surface_img.shape}")
            print(f"   Volume image shape: {volume_img.shape}")
            
            # Convert to PIL and save
            surf_array = (np.clip(surface_img, 0, 1) * 255).astype(np.uint8)
            vol_array = (np.clip(volume_img, 0, 1) * 255).astype(np.uint8)
            
            Image.fromarray(surf_array).save(f"{args.output_dir}/surface_render.png")
            Image.fromarray(vol_array).save(f"{args.output_dir}/volume_render.png")
            
            print(f"💾 Saved surface render to {args.output_dir}/surface_render.png")
            print(f"💾 Saved volume render to {args.output_dir}/volume_render.png")
            
            # Compute and print difference statistics
            surf_gray = np.mean(surface_img, axis=-1)
            vol_gray = np.mean(volume_img, axis=-1)
            diff = np.abs(surf_gray - vol_gray)
            
            print(f"🔍 [ANALYSIS] Image difference statistics:")
            print(f"   Mean difference: {np.mean(diff):.6f}")
            print(f"   Max difference: {np.max(diff):.6f}")
            print(f"   Std difference: {np.std(diff):.6f}")
            print(f"   Pixels with >1% diff: {np.sum(diff > 0.01)}/{diff.size} ({100*np.sum(diff > 0.01)/diff.size:.2f}%)")
            
            if np.max(diff) < 1e-6:
                print("❌ CRITICAL: Surface and volume images are essentially IDENTICAL!")
            else:
                print("✅ Surface and volume images are different")
            
        else:
            print("❌ render_dual_separate method not available")
            
    except Exception as e:
        print(f"❌ Dual separate rendering failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🔍 [DEBUG] Debug script completed")

if __name__ == "__main__":
    main() 