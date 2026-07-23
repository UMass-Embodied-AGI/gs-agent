# Code Block: init
import genesis as gs
import os
from PIL import Image
gs.init(backend=gs.gpu, logging_level='info')
options_mapping = {
	'sim_options': gs.options.SimOptions,
	'vis_options': gs.options.VisOptions,
	'coupler_options': gs.options.BaseCouplerOptions,
	'rigid_options': gs.options.RigidOptions,
	'mpm_options': gs.options.MPMOptions,
	'sph_options': gs.options.SPHOptions,
	'fem_options': gs.options.FEMOptions,
	'pbd_options': gs.options.PBDOptions,
	'profiling_options': gs.options.ProfilingOptions,
	'viewer_options': gs.options.ViewerOptions,
}
scene_args = {'show_viewer': False, 'vis_options': {'show_world_frame': False, 'show_link_frame': False, 'show_cameras': False, 'shadow': True, 'plane_reflection': False, 'background_color': [0.1, 0.1, 0.12], 'visualize_mpm_boundary': False, 'visualize_sph_boundary': False, 'visualize_pbd_boundary': False, 'segmentation_level': 'link', 'render_particle_as': 'sphere'}, 'profiling_options': {'show_FPS': True}, 'mpm_options': {'grid_density': 128, 'particle_size': 0.003, 'enable_CPIC': False, 'lower_bound': [-0.6, -0.5, -0.2], 'upper_bound': [0.6, 0.5, 0.8]}, 'sim_options': {'dt': 0.001, 'substeps': 8, 'gravity': [0.0, 0.0, -9.81]}}
renderer = 'RayTracer'
# LIGHTS = [{'pos': [1.5, 0.8, 2.0], 'color': [1.0, 0.92, 0.85], 'intensity': 3.0, 'radius': 0.6}, {'pos': [-1.2, -1.0, 1.6], 'color': [0.9, 0.95, 1.0], 'intensity': 1.2, 'radius': 0.5}]
# env_sphere = {'image_path': 'demo/brown_photostudio_02_4k.exr', 'image_color': [0.85, 0.9, 0.95], 'radius': 1000.0, 'pos': [0.0, 0.0, 0.0], 'euler': [0.0, 0.0, 0.0]}

is_particle_view = False
# is_particle_view = True

LIGHTS = [{'pos': [-6.0, -5.5, 3.5], 'color': [1.0, 0.92, 0.82], 'intensity': 30.0, 'radius': 3.2}]
env_sphere = {'image_path': 'demo/desert.exr', 'image_color': [0.75, 0.85, 0.95], 'radius': 1000.0, 'pos': [0.0, 0.0, 0.0], 'euler': [0.0, 0.0, 0.0]}
if renderer == 'RayTracer':
	env_sphere_args = {
		'env_pos': env_sphere.get('pos', (0.0, 0.0, 0.0)),
		'env_radius': env_sphere.get('radius', 1000.0),
		'env_euler': env_sphere.get('euler', (0.0, 0.0, 0.0)),
		'env_surface': gs.surfaces.Emission(
			emissive_texture=gs.textures.ImageTexture(
				image_path=env_sphere.get('image_path'),
				image_color=env_sphere.get('image_color', (0.6, 0.6, 0.6)),
			)
		)
	} if env_sphere else {}
	if is_particle_view:
		scene_args['renderer'] = gs.options.renderers.RayTracer(
				lights=LIGHTS,
			)
		camera_args = {'res': [1280, 720], 'pos': [-0.33, 0.12, 0.09], 'lookat': [-0.25, 0.0, 0.06], 'up': [0, 0, 1], 'fov': 55, 'model': 'pinhole', 'spp': 256}
	else:
		scene_args['renderer'] = gs.options.renderers.RayTracer(
				lights=LIGHTS,
				**env_sphere_args
			)
		camera_args = {'res': [1280, 720], 'pos': [-0.33, 0.12, 0.09], 'lookat': [-0.25, 0.0, 0.06], 'up': [0, 0, 1], 'fov': 55, 'model': 'thinlens', 'aperture': 3.0, 'spp': 256}
else:
	scene_args['renderer'] = gs.options.renderers.Rasterizer()
	scene_args['vis_options']['lights'] = LIGHTS
for key, value in scene_args.items():
	if isinstance(value, dict):
		scene_args[key] = options_mapping[key](**value)
	else:
		scene_args[key] = value
scene = gs.Scene(**scene_args)
camera = scene.add_camera(**camera_args)
coacd_options=gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
log_dir = 'logs/reproduce_log'

def get_entity_info(entity_idx: int) -> dict:
    """Gathers and returns information about a specific entity."""
    entity: gs.Entity = scene.entities[entity_idx]
    if isinstance(entity, gs.engine.entities.RigidEntity):
        entity_info = {
            "pos": entity.get_pos().cpu().numpy().tolist(),
            "vel": entity.get_vel().cpu().numpy().tolist(),
        }
    elif isinstance(entity, gs.engine.entities.ParticleEntity): # MPM Entity
        entity_info = {
            "pos": entity.get_particles_pos().detach().cpu().numpy()[0].tolist(),
            "vel": entity.get_particles_vel().detach().cpu().numpy()[0].tolist(),
        }
    else:
        raise ValueError(f"Entity type {type(entity).__name__} not supported.")
    return entity_info

def camera_control(camera, entity_name2idx: dict, current_scene_step: int, dt: float, horizon: int, is_lst_step: bool, video_save_to_filename: str):
    import os, math
    from PIL import Image

    # Timing
    t = current_scene_step * dt
    total_time = horizon * dt

    # Start recording at step 0
    if current_scene_step == 0:
        camera.start_recording()
        os.makedirs(video_save_to_filename.replace('.mp4', ''), exist_ok=True)

    # Helper: smoothstep for transitions
    def smoothstep(a, b, s):
        s = max(0.0, min(1.0, s))
        s = s * s * (3 - 2 * s)
        return a * (1 - s) + b * s

    # Phase A: Macro on sand near predicted impact (t in [0,1] s)
    A_start, A_end = 0.0, 1.0
    macro_pos = [-0.33, 0.12, 0.09]
    macro_look = [-0.25, 0.0, 0.06]

    # Phase B: Follow from behind-right-above (t in [1, total])
    B_start = 1.0

    # Get ball state (if available)
    ball_idx = entity_name2idx.get('basketball', None)
    if ball_idx is not None:
        ball_info = get_entity_info(ball_idx)
        bx, by, bz = ball_info['pos']
    else:
        # Fallback if mapping is missing
        bx, by, bz = (-0.25, 0.0, 0.2)

    # Define follow camera target relative to ball (assumes roll along +x)
    follow_offset = [-0.28, 0.18, 0.16]  # behind (-x), right (+y), above (+z)
    target_pos = [bx + follow_offset[0], by + follow_offset[1], bz + follow_offset[2]]
    # Look slightly ahead and down to the contact patch/rut
    target_look = [bx + 0.06, by + 0.00, bz - 0.12]

    # Blend macro to follow around t=1s
    if t <= A_end:
        # Optionally add a subtle dolly-in during macro
        s = (t - A_start) / max(1e-6, (A_end - A_start))
        # Move a few centimeters closer and lower to emphasize relief
        dolly_pos = [macro_pos[0] - 0.015 * s, macro_pos[1], macro_pos[2] - 0.005 * s]
        dolly_look = macro_look
        cam_pos = dolly_pos
        cam_look = dolly_look
    else:
        # Smoothly transition from macro pose to follow pose over ~0.6s
        trans_dur = 0.6
        s = (t - B_start) / trans_dur
        s = max(0.0, min(1.0, s))
        cam_pos = [
            smoothstep(macro_pos[0], target_pos[0], s),
            smoothstep(macro_pos[1], target_pos[1], s),
            smoothstep(macro_pos[2], target_pos[2], s),
        ]
        cam_look = [
            smoothstep(macro_look[0], target_look[0], s),
            smoothstep(macro_look[1], target_look[1], s),
            smoothstep(macro_look[2], target_look[2], s),
        ]
        # After transition, keep following the ball closely
        if s >= 1.0:
            cam_pos = target_pos
            cam_look = target_look

    # Apply camera pose
    camera.set_pose(pos=cam_pos, lookat=cam_look)

    # Render cadence for ~30 FPS
    target_fps = 30
    render_frequency = max(1, int(1 / (dt * target_fps)))
    print(f"Render frequency: every {render_frequency} steps")

    if current_scene_step % render_frequency == 0:
        rgb, _, segmentation, normal = camera.render(rgb=True,depth=True,segmentation=True,colorize_seg=True,normal=True,antialiasing=True,force_render=True)

        folder = video_save_to_filename.replace('.mp4', '')
        os.makedirs(folder, exist_ok=True)
        frame_idx = current_scene_step // render_frequency

        if is_particle_view:
            particle_path = os.path.join(folder, f'particle_{frame_idx:05d}.png')
            Image.fromarray(rgb).save(particle_path)
        else:
            rgb_path = os.path.join(folder, f'rgb_{frame_idx:05d}.png')
            segmentation_path = os.path.join(folder, f'segmentation_{frame_idx:05d}.png')
            normal_path = os.path.join(folder, f'normal_{frame_idx:05d}.png')
            Image.fromarray(segmentation).save(segmentation_path)
            Image.fromarray(normal).save(normal_path)
            Image.fromarray(rgb).save(rgb_path)


    # Save video at end with FPS matching sim duration
    if is_lst_step:
        sim_duration = max(dt, total_time)
        fps = max(1, int(len(camera._recorded_imgs) / sim_duration))
        camera.stop_recording(save_to_filename=video_save_to_filename, fps=fps)


def entities_control(entities: list, entity_name2idx: dict, emitters: list, emitter_name2idx: dict, current_scene_step: int, horizon: int):
    # Initialize particle bed and basketball initial conditions at step 0 only
    if current_scene_step == 0:
        # Ensure sand starts quiescent
        if 'sand_bed' in entity_name2idx:
            try:
                entities[entity_name2idx['sand_bed']].set_velocity([0.0, 0.0, 0.0])
            except Exception as e:
                print(f"Warning: could not set initial velocity for sand_bed: {e}")
        # Set initial linear and angular velocity for the basketball
        if 'basketball' in entity_name2idx:
            try:
                # DOF velocities: [vx, vy, vz, wx, wy, wz]
                entities[entity_name2idx['basketball']].set_dofs_velocity([1.2, 0.0, 0.0, 0.0, 5.0, 0.0])
            except Exception as e:
                print(f"Warning: could not set initial velocity for basketball: {e}")
    # No active control thereafter; system evolves under physics.
    return

# Code Block: ground_plane
# ground_plane = scene.add_entity(
#     morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), fixed=True),
#     material=gs.materials.Rigid(),
#     surface=gs.surfaces.Default(color=(0.92, 0.92, 0.92), roughness=0.6, double_sided=True)
# )
# Code Block: sand_bed
if is_particle_view:
    sand_bed = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.055-0.08), euler=(0.0, 0.0, 0.0), size=(1.02, 0.8, 0.06+0.14), fixed=False),
        material=gs.materials.MPM.Sand(rho=1600.0, sampler='random', friction_angle=42),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6), vis_mode='particle')
    )
    # Code Block: basketball
    basketball = scene.add_entity(
        morph=gs.morphs.Mesh(coacd_options=coacd_options, file='demo/ball.glb', scale=1.0, pos=(-0.50, 0.00, 0.60), euler=(90.0, 0.0, 90.0), fixed=False),
        material=gs.materials.Rigid(rho=1690.0, friction=None, coup_friction=0.7, coup_softness=0.002, coup_restitution=0.05),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6))
    )
else:
    sand_bed = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.055-0.08), euler=(0.0, 0.0, 0.0), size=(1.02, 0.8, 0.06+0.14), fixed=False),
        material=gs.materials.MPM.Sand(rho=1600.0, sampler='random', friction_angle=42),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.55), double_sided=True, vis_mode='particle')
    )
    # Code Block: basketball
    basketball = scene.add_entity(
        morph=gs.morphs.Mesh(coacd_options=coacd_options, file='demo/ball.glb', scale=1.0, pos=(-0.50, 0.00, 0.60), euler=(90.0, 0.0, 90.0), fixed=False),
        material=gs.materials.Rigid(rho=1690.0, friction=None, coup_friction=0.7, coup_softness=0.002, coup_restitution=0.05),
        surface=gs.surfaces.Rough(double_sided=True)
    )


# Code Block: advance_scene

if not scene.is_built:
    scene.build()
horizon = 2400 # 6000 
os.makedirs("demo_code/ball", exist_ok=True)
for _ in range(horizon):
    gs.logger.info(f"{_}/{horizon} steps advanced")
    entities_control(scene.entities, {'sand_bed': 0, 'basketball': 1}, scene.emitters, {}, scene.t, horizon)
    camera_control(camera, {'sand_bed': 0, 'basketball': 1}, scene.t, scene.dt, horizon, _ == horizon - 1, "demo_code/ball.mp4")
    scene.step()

