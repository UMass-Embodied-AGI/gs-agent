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
scene_args = {'show_viewer': False, 'vis_options': {'show_world_frame': False}, 'profiling_options': {'show_FPS': True}, 'mpm_options': {'grid_density': 128, 'enable_CPIC': True, 'lower_bound': [-1.0, -1.0, 0.0], 'upper_bound': [1.0, 1.0, 1.5]}, 'sim_options': {'dt': 0.0006, 'substeps': 10, 'gravity': [0.0, 0.0, -9.81]}}
camera_args = {'res': [1280, 720], 'pos': [0.4, -0.4, 0.95], 'lookat': [0.0, 0.0, 0.76], 'up': [0.0, 0.0, 1.0], 'fov': 28, 'model': 'pinhole', 'aperture': 2.8, 'spp': 1024}
LIGHTS = [{'pos': [-3.0, -2.0, 5.0], 'color': [1.0, 0.98, 0.95], 'intensity': 6, 'radius': 1.5}, {'pos': [3.0, 2.0, 5.0], 'color': [1.0, 1.0, 1.0], 'intensity': 5, 'radius': 1.5}, {'pos': [0.0, -4.0, 3.5], 'color': [0.95, 1.0, 1.05], 'intensity': 4, 'radius': 1.2}]
renderer = 'RayTracer'
env_sphere = {'image_path': 'assets/hdris/brown_photostudio_02.exr', 'image_color': [0.6, 0.65, 0.7], 'radius': 1000.0, 'pos': [0.0, 0.0, 0.0], 'euler': [0.0, 0.0, -35.0]}
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
    # Macro close-up RayTracer recorder: start on first invocation, run 14168 steps (~8.5 s), fixed macro pose
    import os
    from PIL import Image

    # Fixed macro pose every step (raised z to ensure full plate edge + dough are in frame)
    camera.set_pose(pos=[0.4, -0.4, 0.95], lookat=[0.0, 0.0, 0.76])

    # Persistent state
    if not hasattr(camera, '_sc_macro_started'):
        camera._sc_macro_started = False
        camera._sc_macro_start_step = 0

    # Start recording on first callback
    if not camera._sc_macro_started:
        camera.start_recording()
        camera._sc_macro_started = True
        camera._sc_macro_start_step = current_scene_step

    steps_elapsed = current_scene_step - camera._sc_macro_start_step

    # Fixed cadence for ~30 FPS at dt=0.0006
    render_frequency = 55

    if steps_elapsed >= 0 and steps_elapsed <= 14168 and (steps_elapsed % render_frequency == 0):
        camera.render()
        # Save each frame for traceability
        image_dir = video_save_to_filename.replace('.mp4', '')
        os.makedirs(image_dir, exist_ok=True)
        img_idx = len(camera._recorded_imgs) - 1
        Image.fromarray(camera._recorded_imgs[-1]).save(os.path.join(image_dir, f'frame_{img_idx}.png'))

    # Stop after step budget or at last step
    if camera._sc_macro_started and (steps_elapsed >= 14168 or is_lst_step):
        fps = max(1, int(round(len(camera._recorded_imgs) / 8.5)))
        camera.stop_recording(save_to_filename=video_save_to_filename, fps=fps)


def entities_control(entities: list, entity_name2idx: dict, emitters: list, emitter_name2idx: dict, current_scene_step: int, horizon: int):
    dt = 0.001
    t = current_scene_step * dt

    gp = entities[entity_name2idx['glass_plate']]

    if current_scene_step == 0:
        gp.set_dofs_kp([15000.0, 15000.0, 20000.0, 1000.0, 1000.0, 1000.0])
        gp.set_dofs_kv([300.0, 300.0, 400.0, 50.0, 50.0, 50.0])

    z0 = 0.95
    z1 = 0.7565  # bottom face at 0.7515 m (1.5 mm above tabletop)

    if t <= 3.0:
        alpha = t / 3.0
        z_tgt = z0 + (z1 - z0) * alpha
    else:
        # Hold from 3.0 s to 8.0 s and beyond
        z_tgt = z1

    gp.control_dofs_position([0.0, 0.0, z_tgt, 0.0, 0.0, 0.0])

    return

# Code Block: ground
ground = scene.add_entity(
    morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), fixed=True),
    material=gs.materials.Rigid(rho=200.0, friction=1.0, coup_friction=0.6, coup_softness=0.002, coup_restitution=0.0),
    surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5, ior=1.5)
)
# Code Block: workbench
workbench = scene.add_entity(
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.72), euler=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.06), fixed=True),
    material=gs.materials.Rigid(rho=600.0, friction=0.10, coup_friction=0.10, coup_softness=0.002, coup_restitution=0.0),
    surface=gs.surfaces.Rough(color=(0.60, 0.42, 0.25), double_sided=True)
)
# Code Block: glass_plate
glass_plate = scene.add_entity(
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.95), euler=(0.0, 0.0, 0.0), size=(0.30, 0.30, 0.01), fixed=False),
    material=gs.materials.Rigid(rho=2500.0, friction=0.6, coup_friction=0.05, coup_softness=0.003, coup_restitution=0.0),
    surface=gs.surfaces.Glass(color=(0.9, 0.95, 1.0), double_sided=True)
)
# Code Block: dough
dough = scene.add_entity(
    morph=gs.morphs.Cylinder(pos=(0.0, 0.0, 0.785), euler=(0.0, 0.0, 0.0), radius=0.10, height=0.06, fixed=False),
    material=gs.materials.MPM.ElastoPlastic(E=1.5e4, nu=0.47, rho=1000.0, sampler='pbs', von_mises_yield_stress=7.0e2),
    surface=gs.surfaces.Default(color=(0.97, 0.94, 0.76), roughness=0.6, ior=1.4, vis_mode='recon')
)
# Code Block: reset_scene

if scene.is_built:
    scene.reset()

# Code Block: advance_scene

if not scene.is_built:
    scene.build()
horizon = 14168
os.makedirs("logs/reproduce_log/rendered_video_20251028-163059", exist_ok=True)
for _ in range(horizon):
    gs.logger.info(f"{_}/{horizon} steps advanced")
    entities_control(scene.entities, {'ground': 0, 'workbench': 1, 'glass_plate': 2, 'dough': 3}, scene.emitters, {}, scene.t, horizon)
    camera_control(camera, {'ground': 0, 'workbench': 1, 'glass_plate': 2, 'dough': 3}, scene.t, scene.dt, horizon, _ == horizon - 1, "logs/reproduce_log/rendered_video_20251028-163059.mp4")
    scene.step()

