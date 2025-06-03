import numpy as np
import os
import torch
import multiprocessing
import shelve
from omegaconf import OmegaConf

from sam_maps.models.RS.rs import RS
from sam_maps.models.RS.utils.utils import distribute_work
from sam_maps.models.RS.auto_rs.utils.road_annotation_generator import AutomaticRoadMaskGenerator, GraphRoadGenerator


class AutoRS(RS):
    def __init__(self, config):
        super().__init__(config)


    def forward(self, inputs, graph):
        save_dir = os.path.join(self.config.pred_datapath, self.config.experiment_name, "RS")
        os.makedirs(save_dir, exist_ok=True)
        if self.config.rs.method == "graph":
            crs = self.config.target_crs
            # Expand the centerlines by w_init
            graph_road_generator = GraphRoadGenerator(self.config)
            lane_dict = graph_road_generator.generate_lanes(graph)

        elif self.config.rs.method == "segmentation":
            if not torch.cuda.is_available():
                raise RuntimeError("No GPUs are available.")
        
            processes_per_gpu = self.config.processes_per_gpu
            gpus = self.config.gpus
            if len(gpus) > torch.cuda.device_count():
                raise ValueError(f"{len(gpus)} GPUs were selected, while only {torch.cuda.device_count()} GPUs were available")
            
            num_workers = processes_per_gpu * len(gpus)
            multiprocessing.set_start_method("spawn", force=True)


            cache_folder = os.path.join(self.config.cache_datapath, self.config.experiment_name)
            if not os.path.isdir(cache_folder):
                os.makedirs(cache_folder)
            use_cache = self.config.flags.use_cache
            save_cache = self.config.flags.save_cache
            lane_dict = {}
            if use_cache:
                files = np.unique([".".join(f.split(".")[:-1]) for f in os.listdir(cache_folder)])
                for file in files:
                    cache_file = os.path.join(cache_folder, file)
                    with shelve.open(cache_file, flag="r") as db:
                        cache_dict = {key: db[key] for key in db}
                        lane_dict.update(cache_dict)
            else:
                # Clear cache folder
                for file in os.listdir(cache_folder):
                    os.remove(os.path.join(cache_folder, file))

            missing_lane_keys = sorted(list(set(list(graph["original_edges"].keys())) -  set(list(lane_dict.keys()))), key=int)

            if len(missing_lane_keys)!=0:
                tasks_division = distribute_work(missing_lane_keys, num_workers)

                inputs = [(OmegaConf.to_container(self.config, resolve=True), graph, gpus[i//processes_per_gpu], i, tasks_division[i], os.path.join(cache_folder, f"_{i}.shlv"), save_cache, use_cache) for i in range(num_workers)]
                # profiler = cProfile.Profile()
                # profiler.enable()

                with multiprocessing.Pool(processes=num_workers) as pool:
                    results = pool.starmap(self._initiate_road_annotation_worker, inputs)

                for result in results:
                    lane_dict.update(result)

            # AutomaticRoadMaskGenerator.finalize_map_generation(graph, lane_dict, save_dir) Road Connection
            
        else:
            raise ValueError("Invalid method inserted")
        return lane_dict

    @staticmethod
    def _initiate_road_annotation_worker(config: str, network_data: dict, gpu_id: int, process_id: str, lane_keys: list, cache_file: str, save_cache: bool, use_cache: bool) -> dict:
        config = OmegaConf.create(config)
        road_mask_generator = AutomaticRoadMaskGenerator(config, gpu_id, process_id)
        result = road_mask_generator.generate_lanes(network_data, lane_keys, cache_file=cache_file, save_cache=save_cache, use_cache=use_cache)
        return result
        