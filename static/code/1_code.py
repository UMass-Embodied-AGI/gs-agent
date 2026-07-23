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
scene_args = {'show_viewer': False, 'vis_options': {'show_world_frame': False}, 'profiling_options': {'show_FPS': True}, 'sim_options': {'dt': 0.0008, 'substeps': 10, 'gravity': [0, 0, -9.81]}, 'mpm_options': {'grid_density': 256, 'particle_size': 0.002, 'enable_CPIC': True, 'lower_bound': [-0.4, -0.4, 0.0], 'upper_bound': [0.4, 0.4, 0.5]},}
camera_args = {'res': [1280, 720], 'pos': [0.16, 0.14, 0.18], 'lookat': [0.0, 0.0, 0.118], 'up': [0, 0, 1], 'fov': 34, 'model': 'pinhole', 'spp': 256, 'near': 0.001}
LIGHTS = [{'pos': [0.28, -0.08, 0.23], 'color': [1.0, 0.92, 0.85], 'intensity': 45.0, 'radius': 0.06}, {'pos': [-0.22, 0.18, 0.25], 'color': [1.0, 0.95, 0.9], 'intensity': 22.0, 'radius': 0.04}, {'pos': [0.05, 0.35, 0.16], 'color': [1.0, 0.98, 0.95], 'intensity': 8.0, 'radius': 0.05}]
renderer = 'RayTracer'
env_sphere = {'image_path': 'demo/brown_photostudio_02_4k.exr', 'image_color': [0.7, 0.6, 0.55], 'radius': 1000.0, 'pos': [0.0, 0.0, 0.0], 'euler': [0, 0, 0]}

# is_particle_view = True
is_particle_view = False

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
	else:
		scene_args['renderer'] = gs.options.renderers.RayTracer(
				lights=LIGHTS,
				**env_sphere_args
			)
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
    # Final cinematic: 6.0 s macro close-up with shallow DOF, slow push-in + gentle arc.
    # Emitter remains out of frame; focus on drizzle impact region.
    import math, os
    from PIL import Image

    # Start recording and prep output dir
    if current_scene_step == 0:
        # camera.start_recording()
        os.makedirs(video_save_to_filename.replace('.mp4', ''), exist_ok=True)

    # Choose cinematic fps
    target_fps = 24
    render_frequency = 15 #max(1, int(1.0 / (dt * target_fps)))
    print('render_frequency', render_frequency)

    if current_scene_step % render_frequency == 0:
        # Time phase in [0,1]
        phase = current_scene_step / max(1, horizon - 1)

        # Macro composition around the impact point near z≈0.118
        lookat = [0.0, 0.0, 0.118]

        # Starting camera position (tight macro, lower 3–5 cm of stream visible)
        start_pos = [0.16, 0.14, 0.18]
        offset = [start_pos[i] - lookat[i] for i in range(3)]

        # Slow push-in amount (≈18% closer over the shot)
        push_factor = 1.0 - 0.18 * phase

        # Gentle right-to-left arc (−10° over the shot) around Z axis
        theta = math.radians(-10.0 * phase)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x, y = offset[0], offset[1]
        x_r = cos_t * x - sin_t * y
        y_r = sin_t * x + cos_t * y

        # Slight vertical drift for parallax depth
        z_r = offset[2] + 0.004 * math.sin(2 * math.pi * phase)

        pos = [lookat[0] + push_factor * x_r,
               lookat[1] + push_factor * y_r,
               lookat[2] + push_factor * z_r]

        camera.set_pose(pos=pos, lookat=lookat)

        # Render and save the current frame
        rgb, _, segmentation, normal = camera.render(rgb=True,depth=True,segmentation=True,colorize_seg=True,normal=True,antialiasing=True,force_render=True)

        folder = video_save_to_filename.replace('.mp4', '')
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
        
    # Stop and export at the end; fps matches actual duration
    if is_lst_step:
        fps = max(1, len(camera._recorded_imgs) // int(horizon * dt))
        # camera.stop_recording(save_to_filename=video_save_to_filename, fps=fps)

def entities_control(entities: list, entity_name2idx: dict, emitters: list, emitter_name2idx: dict, current_scene_step: int, horizon: int):
    import math
    # Time parameters for final shot
    dt = 0.0008
    t_start = 0.30
    t_end = 4.50
    start_step = int(t_start / dt)
    end_step = int(t_end / dt)

    # Stream parameters (slow, continuous ribbon)
    nozzle_radius = 0.025  # m
    speed = 0.40  # m/s

    # Gentle circular dither around cake center
    dither_radius = 0.015  # m
    dither_freq = 0.5  # Hz

    if start_step <= current_scene_step <= end_step:
        t = current_scene_step * dt
        # Start phase at t_start so motion begins centered at emission onset
        phase = 2.0 * math.pi * dither_freq * max(0.0, t - t_start)
        dx = dither_radius * math.cos(phase)
        dy = dither_radius * math.sin(phase)
        emitter = emitters[emitter_name2idx['chocolate_emitter']]
        emitter.emit(
            droplet_shape='circle',
            droplet_size=nozzle_radius,
            pos=(dx, dy, 0.22),
            direction=(0.0, 0.0, -1.0),
            theta=0.0,
            speed=speed
        )

    return


if is_particle_view:
    # Code Block: ground
    ground = scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, -1.0), euler=(0.0, 0.0, 0.0), fixed=True),
        material=gs.materials.Rigid(rho=200.0, friction=1.0, coup_friction=0.5, coup_softness=0.002, coup_restitution=0.0),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6))
    )
    # Code Block: tabletop
    tabletop = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.025), euler=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.05), fixed=True),
        material=gs.materials.Rigid(rho=600.0, friction=1.2, coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6))
    )
    # Code Block: plate
    # Increase coupling adhesion parameters on the plate to prevent splashes and promote sticking
    s = 0.26 / (0.1302 - (-0.1302))
    new_pos_z = (-(-0.0085) * s) + 0.0007
    plate = scene.add_entity(
        morph=gs.morphs.Mesh(coacd_options=coacd_options, file='demo/plate.glb', scale=s, pos=(0.0, 0.0, new_pos_z), euler=(90, 0, 90), fixed=False),
        material=gs.materials.Rigid(rho=800.0, friction=0.9, coup_friction=1.5, coup_softness=0.006, coup_restitution=0.0),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6))
    )
    # Code Block: cake
    cake = scene.add_entity(
        morph=gs.morphs.Cylinder(pos=(0.0, 0.0, 0.0600), euler=(0.0, 0.0, 0.0), radius=0.09, height=0.10, fixed=False),
        material=gs.materials.Rigid(rho=300.0, friction=1.0, coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0, sdf_min_res=256, sdf_max_res=256),
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6))
    )
    # Code Block: emitter_choc_emitter
    chocolate_emitter = scene.add_emitter(
        material=gs.materials.MPM.Liquid(E=1e6, nu=0.3, rho=1200.0, viscous=True, sampler='random'),
        max_particles=150000,
        surface=gs.surfaces.Rough(color=(0.6, 0.6, 0.6), vis_mode='particle')
    )
else:
    # Code Block: ground
    ground = scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, -1.0), euler=(0.0, 0.0, 0.0), fixed=True),
        material=gs.materials.Rigid(rho=200.0, friction=1.0, coup_friction=0.5, coup_softness=0.002, coup_restitution=0.0),
        surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.8, ior=1.5)
    )
    # Code Block: tabletop
    tabletop = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.025), euler=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.05), fixed=True),
        material=gs.materials.Rigid(rho=600.0, friction=1.2, coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0),
        surface=gs.surfaces.Rough(color=(0.58, 0.40, 0.20), double_sided=True, vis_mode='visual')
    )
    # Code Block: plate
    # Increase coupling adhesion parameters on the plate to prevent splashes and promote sticking
    s = 0.26 / (0.1302 - (-0.1302))
    new_pos_z = (-(-0.0085) * s) + 0.0007
    plate = scene.add_entity(
        morph=gs.morphs.Mesh(coacd_options=coacd_options, file='demo/plate.glb', scale=s, pos=(0.0, 0.0, new_pos_z), euler=(90, 0, 90), fixed=False),
        material=gs.materials.Rigid(rho=800.0, friction=0.9, coup_friction=1.5, coup_softness=0.006, coup_restitution=0.0),
        surface=gs.surfaces.Smooth(double_sided=True, vis_mode='visual')
    )
    # Code Block: cake
    cake = scene.add_entity(
        morph=gs.morphs.Cylinder(pos=(0.0, 0.0, 0.0600), euler=(0.0, 0.0, 0.0), radius=0.09, height=0.10, fixed=False),
        material=gs.materials.Rigid(rho=300.0, friction=1.0, coup_friction=0.8, coup_softness=0.002, coup_restitution=0.0, sdf_min_res=256, sdf_max_res=256),
        surface=gs.surfaces.Default(color=(0.78, 0.62, 0.32), roughness=0.85, ior=1.4)
    )
    # Code Block: emitter_choc_emitter
    chocolate_emitter = scene.add_emitter(
        material=gs.materials.MPM.Liquid(E=1e6, nu=0.3, rho=1200.0, viscous=True, sampler='random'),
        max_particles=150000,
        surface=gs.surfaces.Smooth(color=(0.10, 0.05, 0.03), double_sided=True, vis_mode='recon', recon_backend='splashsurf-1.5-smooth-25')
        # surface=gs.surfaces.Default(color=(0.227, 0.122, 0.059), roughness=0.05, ior=1.4, vis_mode='recon')
    )


# Code Block: advance_scene

if not scene.is_built:
    scene.build()
horizon = 3000
os.makedirs("demo_code/choco", exist_ok=True)
for _ in range(horizon):
    gs.logger.info(f"{_}/{horizon} steps advanced")
    entities_control(scene.entities, {'ground': 0, 'tabletop': 1, 'plate': 2, 'cake': 3}, scene.emitters, {'chocolate_emitter': 0}, scene.t, horizon)
    camera_control(camera, {'ground': 0, 'tabletop': 1, 'plate': 2, 'cake': 3}, scene.t, scene.dt, horizon, _ == horizon - 1, "demo_code/choco.mp4")
    scene.step()

