#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent

# target_directory = 'vit/vtab'
target_directory = 'vit/cifar_100'
# target_directory = 'vit/imagenet_1k'

cuda_devices = [5, 6]

# tuning_method = None
# yuequ_method = None
# pretrained_model = 'vit_tiny_patch16_224_in21k'

# tuning_method = 'ssf'
# yuequ_method = None


# tuning_method = None
# yuequ_method = 'wave_high_ladder'
# yuequ_method = 'wave_high_shoulder'

# force_method = True
force_method = False
if force_method:
    method_name_postfix = ''
    # method_name_postfix = '_blend'
    # method_name_postfix = '_0blend'
    # method_name_postfix = '_pretrain'

    if tuning_method is not None and yuequ_method is not None:
        method_name = f'{tuning_method}-{yuequ_method}'
    elif tuning_method is not None and yuequ_method is None:
        method_name = tuning_method
    elif tuning_method is None and yuequ_method is not None:
        method_name = yuequ_method
    else:
        method_name = 'full_finetune'
        
    method_name += method_name_postfix
else:
    method_name = 'auto'
plan_name = method_name+'-'+target_directory.replace('/', '_')

# params_rate = 11511891/208900
params_rate = 1

#%%

# 等待nvidia-smi存在大于16GiB显存的显卡存在两个，选择那两个卡号运行
import torch
import torch.cuda as cuda
import time
device_count = torch.cuda.device_count()
print(f"device count: {device_count}")
ok_gpus = []

import gpustat
while len(ok_gpus)<2:
    ok_gpus = []
    free_memories = []
    # free_memories = torch.cuda.mem_get_info()
    # print(free_memories)
    gpu_stats = gpustat.GPUStatCollection.new_query()
    print(gpu_stats)
    
    for i in range(device_count):
        # print(f"device {i}: {torch.cuda.get_device_name(i)}")
        # 使用 torch.cuda
        # props = torch.cuda.get_device_properties(i)
        # print(props)
        # free_memory = torch.cuda.memory_allocated(i)

        
        free_memory = gpu_stats[i].memory_free
        # print(free_memory)
        
        free_memories.append(free_memory)
        
        # if free_memory > 16*1024**3:
        # if free_memory > 16*1024:
        if free_memory > 10*1024:
            ok_gpus.append(i)
    if len(ok_gpus)<2:
        print("waiting for more free memory...")
        print('free_memories:', free_memories)
        time.sleep(10)

# choose = 1
choose = 2

ok_gpus.sort(key=lambda x: free_memories[x])
cuda_devices = ok_gpus[-choose:]
print(f"using gpus {cuda_devices}")
         
     
#%%


import json

path_to_vtab_1k = Path('~/datasets/peft/vtab-1k').expanduser()
path_to_cifar100 = Path('~/datasets/peft/cifar-100-python').expanduser()
path_to_imagenet_1k = Path('~/datasets/peft/ImageNet2012').expanduser()

target_directory = this_directory / target_directory
print(target_directory)

contents = {}
# 递归 '*.*sh'
# for file in target_directory.rglob('*.*sh'):
for file in target_directory.rglob('train*.*sh'):
# for file in target_directory.rglob('*.sh'):
# for file in target_directory.rglob('*'):
    # print(file.name)
    if not force_method:
        method_name = file.name[file.name.index('_')+1:file.name.index('.sh')]
    content = ''
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 把 content 中 CUDA_VISIBLE_DEVICES=5,6,...+空格 替换成 cuda_devices 中书写的
    # 使用正则表达式
    
    import re
    import random
    pattern = re.compile(r'CUDA_VISIBLE_DEVICES=\d+(,\d+)*')
    content = pattern.sub(f'CUDA_VISIBLE_DEVICES={",".join(map(str, cuda_devices))}', content)
    # 修改  --nproc_per_node=2 成 --nproc_per_node=len(cuda_devices)
    pattern = re.compile(r'--nproc_per_node=\d+')
    content = pattern.sub(f'--nproc_per_node={len(cuda_devices)}', content)
    
    # 把 python 改为 /home/ycm/program_files/miniconda3/envs/ssf/bin/python
    pattern = re.compile(r' python')
    content = pattern.sub(' /home/ycm/program_files/miniconda3/envs/ssf/bin/python', content)
    
    # 在 "train.py "后面加上 "--data_dir "
    if not "--data_dir" in content:
        pattern = re.compile(r'train.py')
        content = pattern.sub(f'train.py --data_dir ', content)
    
    # 修改 master_port 为随机端口
    pattern = re.compile(r'--master_port=(\d+)')
    if match := pattern.search(content):
        master_port = int(match.group(1))
        master_port = random.randint(10000, 65535)
        content = pattern.sub(f'--master_port={master_port}', content)

    # 所有与lr 有关的参数，都/params_rate，因为参数量越大，学习率越小
    # 这种参数包括  warmup-lr lr min-lr 就是 可能有lr前面有东西
    def make_lr(lr_name, content):
        pattern = re.compile(r'--'+lr_name+r'\s+(\S+)')
        if match := pattern.search(content):
            lr = float(match.group(1))
            content = pattern.sub(f'--{lr_name} {lr/params_rate:e}', content)
        return content
    content = make_lr('lr', content)
    # content = make_lr('min-lr', content)
    # content = make_lr('warmup-lr', content)
    

    # 把 /path/to/vtab-1k/* + 空格 改为 path_to_vtab_1k
    pattern = re.compile(r'/path/to/vtab-1k/')
    content = pattern.sub(path_to_vtab_1k.as_posix()+"/", content)
    
    # 把 /path/to/cifar100 改为 path_to_cifar100
    pattern = re.compile(r'/path/to/cifar100')
    content = pattern.sub(path_to_cifar100.as_posix()+"/", content)
    
    # 把 /path/to/imagenet_1k 改为 path_to_imagenet_1k
    pattern = re.compile(r'/path/to/imagenet_1k')
    content = pattern.sub(path_to_imagenet_1k.as_posix()+"/", content)
    
    # 把 dataset_download 改为 true

    

    
    # 把 --output */*/* (类似于 "output/vit_base_patch16_224_in21k/vtab/caltech101/ssf" )
    # 的 最后一个/ 后面的内容( 比如 ssf) 改为 method_name 
    # 我们先查找 --output 空格 内容 空格，同事也把内容提取出来
    pattern = re.compile(r'--output\s+(\S+)')
    if match := pattern.search(content):
        output_dir = match.group(1)
        output_dir_without_method = "/".join(output_dir.split('/')[:-1])
        content = pattern.sub(f'--output {output_dir_without_method}/{method_name}', content)
    else:
        raise ValueError(f"can't find --output in {file}")
    

    
     
    # 去掉 \\
    if content.endswith('\\'):
        content = content[:-1]
    
    if force_method:
        # 修改 --tuning-mode 
        pattern = re.compile(r'--tuning-mode\s+(\S+)')
        # 如果 content里面已经存在了，就要覆盖改为我们指定的
        # 如果没有我们就要手动添加
        if pattern.search(content):
            content = pattern.sub(f'--tuning-mode {tuning_method}', content)
        else:
            content += f' --tuning-mode {tuning_method} '
        
        # 修改 --yuequ-method， 逻辑类似
        pattern = re.compile(r'--yuequ-method\s+(\S+)')
        if pattern.search(content):
            content = pattern.sub(f'--yuequ-method {yuequ_method}', content)
        else:
            content += f' --yuequ-method {yuequ_method} '
            
        # 把 --model xxx 改为 --model vit_tiny_patch16_224_in21k
        pattern = re.compile(r'--model\s+(\S+)')
        if match := pattern.search(content):
            model_name = match.group(1)
            content = pattern.sub(f'--model {pretrained_model}', content)
        else:
            content += f' --model {pretrained_model} '
    

    # print(content)
    # assert type(content) == str
    
    contents[file.as_posix()] = content
# print(contents)
    
with open(this_directory / f'plans_{plan_name}.json', 'w') as f:
    json.dump(contents, f, ensure_ascii=False, indent=4)
    

def dump_mem(memory):
    with open(this_directory/(memory_file), 'w') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)
    
    
# 执行 contents，同时如果有错误，打印错误信息退出，如果成功则下次运行不需要再次执行

import json
memory_file = f'auto_run_memory_{plan_name}.json'
if (this_directory/memory_file).exists():
    with open(this_directory/(memory_file), 'r') as f:
        memory = json.load(f)        
else:
    memory = dict()
    dump_mem(memory)

import subprocess    

import os

os.chdir((this_directory/'..').as_posix())

# conda
# check_output = subprocess.check_output('source /home/ycm/program_files/miniconda3/bin/activate ssf', shell=True)

def execute_content(content
                    # , verbose=True
                    )->dict:
    print(f"executing command:\n\t {content}")
    try:
        # 需要把内容实时打印到这个终端，同时得到output，同时需要shell
        # output = subprocess.check_output(content, shell=True, stderr=subprocess.STDOUT)
        # output = subprocess.check_output(content, shell=True, stderr=subprocess.PIPE)
        process = subprocess.Popen(content, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output = ""
        # 实时读取输出
        for line in process.stdout:
            print(line, end='')  # 打印到屏幕
            output += line  # 将输出内容追加到变量

        # 等待命令执行完成
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=content, output=output.encode('utf-8'))
        # if verbose:
        #     output = subprocess.check_output(content, shell=True, stderr=subprocess.STDOUT)
        # else:
        #     output = subprocess.check_output(content, shell=True, stderr=subprocess.DEVNULL)
        return dict(status='done', output=output)
    except subprocess.CalledProcessError as e:
        return dict(status='error', output=e.output.decode('utf-8'))

for file, content in contents.items():
    if file in memory:
        d = memory[file][-1]
        status = d.get('status', 'unknown')
        print(f"file {file} has been executed with status {status}")
        if status == 'error':
            print("retring...")
            
            memory[file].append(dict(status='running', output='retrying...'))
            dump_mem(memory)
            
            new_output = execute_content(content)
            print(f"exited, status is {new_output['status']}")
            memory[file].append(new_output)
            
        elif status == 'done':
            continue
        elif status == 'running':
            print("Other instance of this script is running this task, skipping...")
            continue
        else:
            print("status unknown, changing it to error")
            memory[file].append(dict(status='error'))
    else:
        print(f"executing file {file}")
        memory[file] = [
            dict(status='running', output='executing...')
        ]
        dump_mem(memory)
        new_output = execute_content(content)
        print(f"exited, status is {new_output['status']}")
        memory[file] = [
            new_output
        ]
        
    dump_mem(memory)
    time.sleep(5)
    
    # with open(this_directory/(memory_file), 'w') as f:
    #     json.dump(memory, f, ensure_ascii=False, indent=4)
        


    
# %%
