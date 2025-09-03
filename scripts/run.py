#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

	parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%04d.png).")

	parser.add_argument("--name", type=str, default="", help="Experiment name. Outputs will be organized as dataset/scene/name under the repo root.")
	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--configuration", choices=["baseline","surface","volume","hybrid","dual_separate","dual_merge"], default=None, help="High-level model configuration. If provided and --network is empty, selects configs/nerf/<configuration>.json")
	parser.add_argument("--loss_mode", choices=["baseline","surface","hybrid"], default="baseline", help="Select loss: baseline (composited color), surface (per-sample radiance), hybrid (future use)")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")
	parser.add_argument("--use_sdf", action="store_true", help="If set, treat density MLP's first output as SDF and convert to density for alpha blending.")
	parser.add_argument("--eikonal_lambda", type=float, default=0.0, help="Eikonal loss weight for SDF normals (||\u2207SDF||-1)^2. Default 0.0.")


	return parser.parse_args()

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

if __name__ == "__main__":
	args = parse_args()
	if args.vr: # VR implies having the GUI running at the moment
		args.gui = True

	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

	# Configure loss routing via environment for kernel guard
	if args.loss_mode == "surface":
		os.environ["NGP_SURFACE_SAMPLE_LOSS"] = "1"
	else:
		os.environ.pop("NGP_SURFACE_LOSS", None)

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	for file in args.files:
		scene_info = get_scene(file)
		if scene_info:
			file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
		testbed.load_file(file)

	if args.scene:
		scene_info = get_scene(args.scene)
		if scene_info is not None:
			args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
			if not args.network and "network" in scene_info:
				args.network = scene_info["network"]

		testbed.load_training_data(args.scene)

		# Create output directory structure: dataset/scene/name if args.scene ends with dataset/scene/transforms_train.json
		# Example: /home/.../data/nerf_synthetic/lego/transforms_train.json -> nerf_synthetic/lego/<name>
		if args.name:
			try:
				parts = args.scene.split("/")
				if len(parts) >= 2 and parts[-1] == "transforms_train.json":
					dataset = parts[-2]  # scene
					data_root = parts[-3]  # dataset folder name
					folder_parts = [data_root, dataset]
					if args.configuration:
						folder_parts.append(args.configuration)
					if args.loss_mode:
						folder_parts.append(args.loss_mode)
					folder_parts.append(args.name)
					out_rel = os.path.join(*folder_parts)
					out_abs = os.path.join(ROOT_DIR, out_rel)
					os.makedirs(out_abs, exist_ok=True)
					# Track output directory for downstream saves
					output_dir_abs = out_abs
					# Prepare subfolders commonly used
					for sub in ["checkpoints", "logs", "images", "evaluation", "mesh", "recording"]:
						os.makedirs(os.path.join(out_abs, sub), exist_ok=True)
					print(f"Outputs will be stored under: {out_abs}")
			except Exception as e:
				print(f"Warning: failed to create structured output folders: {e}")

	# Ensure variable exists even if no name provided
	output_dir_abs = locals().get('output_dir_abs', None)
	images_dir = os.path.join(output_dir_abs, "images") if output_dir_abs else None
	logs_dir = os.path.join(output_dir_abs, "logs") if output_dir_abs else None

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window)
		if args.vr:
			testbed.init_vr()


	if args.load_snapshot:
		scene_info = get_scene(args.load_snapshot)
		if scene_info is not None:
			args.load_snapshot = default_snapshot_filename(scene_info)
		testbed.load_snapshot(args.load_snapshot)
	elif args.network:
		testbed.reload_network_from_file(args.network)
	elif args.configuration:
		# Map configuration to default nerf config JSON
		cfg_path = os.path.join(ROOT_DIR, "configs", "nerf", f"{args.configuration}.json")
		if os.path.exists(cfg_path):
			args.network = cfg_path
			testbed.reload_network_from_file(args.network)
		else:
			print(f"Warning: configuration '{args.configuration}' not found at {cfg_path}")

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True

	# Apply SDF mode if requested
	if getattr(args, "use_sdf", False):
		print("[run.py] Enabling SDF mode: interpreting sigma as SDF and converting to density.")
		try:
			testbed.use_sdf = True
			# Set eikonal lambda (defaults to 0.0)
			try:
				testbed.sdf_eikonal_lambda = float(args.eikonal_lambda)
			except Exception:
				pass
		except Exception as e:
			print(f"[run.py] Warning: could not set testbed.use_sdf: {e}")

	network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
	if testbed.mode == ngp.TestbedMode.Sdf:
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Match nerf paper behaviour and train on a fixed bg.
		testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# Fixed checkpoint path (overwritten) if an output directory is present
	checkpoint_path = os.path.join(output_dir_abs, "checkpoints", "latest.msgpack") if output_dir_abs else None

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="steps") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Overwrite checkpoint every 5000 steps
				if checkpoint_path and (testbed.training_step % 5000 == 0) and testbed.training_step > 0:
					try:
						os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
						testbed.save_snapshot(checkpoint_path, False)
					except Exception as e:
						print(f"Warning: failed to save periodic checkpoint: {e}")

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_snapshot:
		os.makedirs(os.path.dirname(args.save_snapshot), exist_ok=True)
		testbed.save_snapshot(args.save_snapshot, False)
	elif output_dir_abs:
		# Overwrite the fixed checkpoint file at the end as well
		final_snap = os.path.join(output_dir_abs, "checkpoints", "latest.msgpack")
		try:
			print(f"Saving final snapshot to {final_snap}")
			testbed.save_snapshot(final_snap, False)
		except Exception as e:
			print(f"Warning: could not save final snapshot: {e}")

	# Auto-evaluation: if scene was given and no explicit test_transforms, try to use dataset test transforms
	if not args.test_transforms and args.scene:
		try:
			scene_dir = os.path.dirname(args.scene)
			candidate = os.path.join(scene_dir, "transforms_test.json")
			if os.path.exists(candidate):
				args.test_transforms = candidate
		except Exception:
			pass

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		# If we have an output folder, stage eval there
		eval_dir = os.path.join(output_dir_abs, "evaluation") if output_dir_abs else os.getcwd()
		os.makedirs(eval_dir, exist_ok=True)

		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [1.0, 1.0, 1.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.render_min_transmittance = 1e-4

		testbed.shall_train = False
		testbed.load_training_data(args.test_transforms)

		testbed.render_with_lens_distortion = True

		with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
			for i in t:
				resolution = testbed.nerf.training.dataset.metadata[i].resolution
				testbed.render_ground_truth = True
				testbed.set_camera_to_training_view(i)
				ref_image = testbed.render(resolution[0], resolution[1], 1, True)
				testbed.render_ground_truth = False
				image = testbed.render(resolution[0], resolution[1], spp, True)

				if i == 0:
					write_image(os.path.join(eval_dir, "ref.png"), ref_image)
					write_image(os.path.join(eval_dir, "out.png"), image)

					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0
					write_image(os.path.join(eval_dir, "diff.png"), diffimg)

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
		try:
			with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
				json.dump({"psnr": psnr, "psnr_avgmse": psnr_avgmse, "ssim": ssim, "minpsnr": minpsnr, "maxpsnr": maxpsnr}, f, indent=2)
		except Exception as e:
			print(f"Warning: could not write metrics.json: {e}")

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		thresh = args.marching_cubes_density_thresh or 2.5
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)

	if ref_transforms:
		# TODO: load lens & screen center from ref_transforms
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]

			if 'transform_matrix' in f:
				cam_matrix = f['transform_matrix']
			elif 'transform_matrix_start' in f:
				cam_matrix = f['transform_matrix_start']
			else:
				raise KeyError("Missing both 'transform_matrix' and 'transform_matrix_start'")
			
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)

		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps
		save_frames = "%" in args.video_output
		start_frame, end_frame = args.video_render_range

		if "tmp" in os.listdir():
			shutil.rmtree("tmp")
		os.makedirs("tmp")

		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			testbed.camera_smoothing = args.video_camera_smoothing

			if start_frame >= 0 and i < start_frame:
				# For camera smoothing and motion blur to work, we cannot just start rendering
				# from middle of the sequence. Instead we render a very small image and discard it
				# for these initial frames.
				# TODO Replace this with a no-op render method once it's available
				frame = testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
				continue
			elif end_frame >= 0 and i > end_frame:
				continue

			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
			if save_frames:
				write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
			else:
				write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

		if not save_frames:
			os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")

		shutil.rmtree("tmp")

	# After training: optional rendering of training views every 25 (train + test)
	if output_dir_abs and testbed.nerf.training.dataset.n_images > 0:
		try:
			os.makedirs(images_dir, exist_ok=True)
			with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Saving training views (stride 25)") as t2:
				for i in t2:
					if i % 25 != 0:
						continue
					res = testbed.nerf.training.dataset.metadata[i].resolution
					testbed.set_camera_to_training_view(i)
					img = testbed.render(res[0], res[1], 8, True)
					write_image(os.path.join(images_dir, f"train_{i:04d}.png"), img)
		except Exception as e:
			print(f"Warning: failed to save stride-25 training views: {e}")

	# Also render test views every 25 if available, saving both GT and rendered
	try:
		if args.test_transforms and os.path.exists(args.test_transforms):
			with open(args.test_transforms) as f:
				_ = json.load(f)
			# Load test set for indexing
			testbed.load_training_data(args.test_transforms)
			with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Saving test views (stride 25)") as t3:
				for i in t3:
					if i % 25 != 0:
						continue
					res = testbed.nerf.training.dataset.metadata[i].resolution
					testbed.set_camera_to_training_view(i)
					
					# Save ground truth image
					testbed.render_ground_truth = True
					gt_img = testbed.render(res[0], res[1], 1, True)
					write_image(os.path.join(images_dir, f"gt_{i:04d}.png"), gt_img)
					
					# Save rendered image
					testbed.render_ground_truth = False
					rendered_img = testbed.render(res[0], res[1], 8, True)
					write_image(os.path.join(images_dir, f"test_{i:04d}.png"), rendered_img)
	except Exception as e:
		print(f"Warning: failed to save stride-25 test views: {e}")

	# Write basic logs (loss progression)
	if output_dir_abs:
		try:
			os.makedirs(logs_dir, exist_ok=True)
			with open(os.path.join(logs_dir, "meta.txt"), "w") as f:
				f.write(f"configuration={args.configuration}\n")
				f.write(f"loss_mode={args.loss_mode}\n")
				f.write(f"use_sdf={getattr(args, 'use_sdf', False)}\n")
		except Exception as e:
			print(f"Warning: could not write logs: {e}")
