import argparse
from ast import Dict
import glob
import json
import os
from threading import Thread
from queue import Queue
from typing import Dict
import tqdm

from dataclasses import dataclass


RUN_SEG_CMD = '/home/nlp/egsotic/repo/HebPipe/run_seg.sh {server} {gpu_i} {input_path} {output_path}'


@dataclass
class Resource:
    server: str
    gpu_i: int


@dataclass
class CmdTask:
    cmd: str
    kwargs: Dict
    
    def __init__(self, cmd: str, **kwargs):
        self.cmd = cmd
        self.kwargs = dict(**kwargs)
    
    def run(self, **kwargs):
        cmd = self.cmd.format(**self.kwargs, **kwargs)
        
        return os.system(cmd)
        
        # p = Popen(cmd, stdout=PIPE, shell=True, universal_newlines=True)
        
        # for stdout_line in iter(p.stdout.readline, ""):
        #     print(stdout_line)
        # p.stdout.close()
        
        # return p.wait()
    
    def is_complete(self):
        return not should_process_file(self.kwargs['input_path'], self.kwargs['output_path'])

    def __repr__(self) -> str:
        return self.cmd.format(**self.kwargs)

def get_files(glob_input_path, pattern_output_path):
    source, target = pattern_output_path.split(' ')
    
    for file_input_path in tqdm.tqdm(glob.glob(glob_input_path)):
        try:
            file_output_path = file_input_path.replace(f'/{source}/', f'/{target}/')
                    
            if should_process_file(file_input_path, file_output_path):            
                os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
                
                yield file_input_path, file_output_path
        except Exception as e:
            print('get_files exception', e)
            

def should_process_file(file_input_path, file_output_path):
    if not os.path.exists(file_output_path):
        return True
    
    input_lines = count_file_lines(file_input_path)
    output_lines = count_file_lines(file_output_path)
    
    return output_lines / input_lines < 0.9

def count_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f_in:
        total_lines = sum(1 for _ in f_in)

    return total_lines

def run_task_wrapper(task, resource, tasks_queue, resources_queue):
    task.kwargs['server'] = resource.server
    task.kwargs['gpu_i'] = resource.gpu_i
    print('running task:', task, 'using resource:', resource)
    
    try:
        task.run()
        
        if task.is_complete():
            print('finished task:', task)
            tasks_queue.task_done()
        else:
            # re-add task
            print('re-adding task:', task)
            task.kwargs['server'] = None
            task.kwargs['gpu_i'] = None
            
            tasks_queue.put(task)
    except Exception as e:
        print('exception at task:', task, e)
        
        # re-add task
        task.kwargs['server'] = None
        task.kwargs['gpu_i'] = None
        
        tasks_queue.put(task)
    finally:
        task.kwargs['server'] = None
        task.kwargs['gpu_i'] = None
        
        resources_queue.put(resource)

def scheduler_loop(tasks, resources, tasks_queue):
    resources_queue = Queue()
    
    for r in resources:
        resources_queue.put(r)
    
    for task in tasks:
        tasks_queue.put(task)
    
    while True:
        print('waiting for next task...')
        task = tasks_queue.get()
        if task is None:
            break
        
        print('waiting for next resource...')
        resource = resources_queue.get()
        
        thread = Thread(target=run_task_wrapper, args=(task, resource, tasks_queue, resources_queue))
        thread.start()

def main(config):    
    resources = [
        Resource(server=r['server'], gpu_i=gpu_i)    
        for r in config['resources']
        for gpu_i in r['gpus_i']
        for _ in range(r['slots'])
    ]
    
    tasks = [
        CmdTask(RUN_SEG_CMD,
                input_path=input_path,
                output_path=output_path)
        for input_path, output_path in get_files(config['glob_input_path'],
                                                 config['pattern_output_path'])
    ]
    
    tasks_queue = Queue()
    
    print(len(tasks), 'tasks', len(resources), 'resources')
    
    scheduler_loop(tasks, resources, tasks_queue)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path")
    
    args = parser.parse_args()
    
    config_path = args.config_path
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    main(config)

