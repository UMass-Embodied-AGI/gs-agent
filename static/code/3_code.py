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
scene_args = {'show_viewer': False, 'vis_options': {'show_world_frame': False}, 'profiling_options': {'show_FPS': True}, 'sim_options': {'dt': 0.0005, 'substeps': 8, 'gravity': [0, 0, -9.81]}, 'sph_options': {'particle_size': 0.002, 'lower_bound': [-0.5, -0.5, 0.0], 'upper_bound': [0.5, 0.5, 1.0]}}
camera_args = {'res': [1920, 1080], 'pos': [0.0, 0.25, 0.5], 'lookat': [0.0, 0.0, 0.5], 'up': [0.0, 0.0, 1.0], 'fov': 55, 'model': 'pinhole', 'aperture': 2.8, 'spp': 1536}
# LIGHTS = [{'pos': [-0.3, 0.8, 0.7], 'color': [1.0, 0.97, 0.93], 'intensity': 90.0, 'radius': 0.08}, {'pos': [1.5, -0.8, 0.55], 'color': [1.0, 1.0, 1.0], 'intensity': 210.0, 'radius': 0.07}, {'pos': [0.2, 0.9, 0.95], 'color': [1.0, 1.0, 1.0], 'intensity': 35.0, 'radius': 0.12}]
LIGHTS = [{'pos': [0.28, -0.08, 0.23], 'color': [1.0, 0.92, 0.85], 'intensity': 45.0, 'radius': 0.06}, {'pos': [-0.22, 0.18, 0.25], 'color': [1.0, 0.95, 0.9], 'intensity': 22.0, 'radius': 0.04}, {'pos': [0.05, 0.35, 0.16], 'color': [1.0, 0.98, 0.95], 'intensity': 8.0, 'radius': 0.05}]
renderer = 'RayTracer'
env_sphere = {'image_path': 'demo/brown_photostudio_02_4k.exr', 'image_color': [0.7, 0.6, 0.55], 'radius': 1000.0, 'pos': [0.0, 0.0, 0.0], 'euler': [0, 0, 0]}
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
    # FINAL cinematic slow-motion render: fixed side-on camera, RayTracer
    # Record first ~0.12 s at 60 fps playback, rendering every simulation step for crisp slow motion
    import os
    from PIL import Image

    record_duration_s = 0.12
    record_steps = max(1, int(record_duration_s / dt + 1e-9))  # ~240 steps for dt=0.0005
    target_fps = 60  # slow-motion playback length: ~240 frames / 60 fps = 4 s

    # Start capture
    if current_scene_step == 0:
        camera.start_recording()

    # Fixed camera pose per spec
    if current_scene_step < record_steps and current_scene_step % 1 == 0:
        pos = [0.00, 0.15, 0.50]
        lookat = [0.00, 0.00, 0.50]
        camera.set_pose(pos=pos, lookat=lookat)
        camera.render()
        # Save individual frame for QA
        dirpath = video_save_to_filename.replace('.mp4', '')
        os.makedirs(dirpath, exist_ok=True)
        image_path = os.path.join(dirpath, f'frame_{current_scene_step:04d}.png')
        Image.fromarray(camera._recorded_imgs[-1]).save(image_path)

    # Stop once window is covered; save at target fps for slow motion
    if current_scene_step == (record_steps - 1):
        camera.stop_recording(save_to_filename=video_save_to_filename, fps=target_fps)


def entities_control(entities: list, entity_name2idx: dict, emitters: list, emitter_name2idx: dict, current_scene_step: int, horizon: int):
    # Droplet: ensure zero initial velocity only at t=0; then free under gravity
    if current_scene_step == 0 and 'droplet' in entity_name2idx:
        entities[entity_name2idx['droplet']].set_velocity([0.0, 0.0, 0.0])

    # Strawberry: set velocity control gains and initial linear/angular velocity
    if 'strawberry' in entity_name2idx:
        ent = entities[entity_name2idx['strawberry']]
        if current_scene_step == 0:
            # Set PD velocity gains for stable tracking
            ent.set_dofs_kv([10.0] * ent.n_dofs)
            # Linear velocity along +x = 2.5 m/s, mild spin about +z (~0.2 rad/s)
            ent.control_dofs_velocity([2.5, 0.0, 0.0, 0.0, 0.0, 0.2])


# Code Block: droplet
droplet = scene.add_entity(
    morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.5), euler=(0.0, 0.0, 0.0), radius=0.015, fixed=False),
    material=gs.materials.SPH.Liquid(rho=1000.0, stiffness=50000.0, exponent=7.0, mu=0.0012, gamma=0.012, sampler='pbs'),
    surface=gs.surfaces.Glass(color=(0.7, 0.85, 1.0), double_sided=True, vis_mode='recon')
)
# Code Block: strawberry
strawberry = scene.add_entity(
    morph=gs.morphs.Mesh(coacd_options=coacd_options, file='demo/strawberry.glb', scale=0.915, pos=(-0.12, 0.0, 0.5), euler=(0.0, 180.0, -90.0), fixed=False),
    material=gs.materials.Rigid(rho=1000.0, friction=0.6, coup_friction=0.1, coup_softness=0.002, coup_restitution=0.0),
    surface=gs.surfaces.Smooth(double_sided=True)
)
# Code Block: reset_scene

if scene.is_built:
    scene.reset()

# Code Block: advance_scene

if not scene.is_built:
    scene.build()
horizon = 240
os.makedirs("demo_code/berry1", exist_ok=True)
for _ in range(horizon):
    gs.logger.info(f"{_}/{horizon} steps advanced")
    entities_control(scene.entities, {'droplet': 0, 'strawberry': 1}, scene.emitters, {}, scene.t, horizon)
    camera_control(camera, {'droplet': 0, 'strawberry': 1}, scene.t, scene.dt, horizon, _ == horizon - 1, "demo_code/berry1.mp4")
    scene.step()

