import os
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List
from types import SimpleNamespace
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from .grasp_generation.utils.hand_model import HandModel
from .grasp_generation.utils.object_model import ObjectModel
from .grasp_generation.utils.initializations import initialize_convex_hull
from .grasp_generation.utils.energy import cal_energy
from .grasp_generation.utils.optimizer import Annealing

dexgraspnet_dir = os.path.dirname(__file__)
mano_dir = os.path.join(dexgraspnet_dir, "grasp_generation", "mano")

def generate(object_path_list: List[str], side: str="right", num_grasps: int=64, seed: int=0) -> List[NDArray[np.float32]]:
    np.seterr(all='raise')
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_model = HandModel(
        mano_root=mano_dir, 
        contact_indices_path=os.path.join(mano_dir, "contact_indices.json"), 
        pose_distrib_path=os.path.join(mano_dir, "pose_distrib.pt" if side=="right" else "pose_distrib_left.pt"), 
        device=device,
        side=side
    )
    object_model = ObjectModel(
        data_root_path=None,
        batch_size_each=num_grasps,
        num_samples=2000, 
        device=device,
    )
    object_model.initialize(object_path_list)
    initialize_convex_hull_args = SimpleNamespace(
        distance_lower=0.1,
        distance_upper=0.1,
        theta_lower=0.,
        theta_upper=0.,
        jitter_strength=0.,
        n_contact=4
    )
    initialize_convex_hull(hand_model, object_model, initialize_convex_hull_args)

    optim_config = {
        'switch_possibility': 0.5,
        'starting_temperature': 18.,
        'temperature_decay': 0.95,
        'annealing_period': 30,
        'step_size': 0.005,
        'stepsize_period': 50,
        'mu': 0.98,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    weight_dict = dict(
        w_dis=100.,
        w_pen=100.,
        w_prior=0.5,
        w_spen=10.
    )
    energy, E_fc, E_dis, E_pen, E_prior, E_spen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)
    energy.sum().backward(retain_graph=True)

    for step in tqdm(range(1, 6000 + 1), desc='optimizing'):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_prior[accept] = new_E_prior[accept]
            E_spen[accept] = new_E_spen[accept]
    
    hand_pose_list: List[NDArray[np.float32]] = []
    for i in range(len(object_path_list)):
        hand_pose = hand_model.hand_pose[i*num_grasps:(i+1)*num_grasps].detach().cpu().numpy()
        hand_pose_list.append(hand_pose)
    return hand_pose_list
