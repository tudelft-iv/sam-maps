def distribute_work(lane_keys, num_workers):
    """
    Split tasks into ranges for parallel workers.
    """
    total_tasks = len(lane_keys)
    chunk_size = (total_tasks // num_workers) 
    if total_tasks % num_workers != 0:
        chunk_size += 1
    task_ranges = [[i * chunk_size, (i + 1) * chunk_size] for i in range(num_workers)]
    if total_tasks % num_workers != 0:
        task_ranges[-1] = [task_ranges[-1][0], total_tasks]
    tasks_division = []
    for task_range in task_ranges:
        tasks_division.append(lane_keys[task_range[0]:task_range[1]])
    return tasks_division
