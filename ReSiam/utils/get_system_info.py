import platform
import psutil  # pip install psutil
import cpuinfo  # pip install py-cpuinfo
import pynvml  # pip install pynvml

from collections import Counter

class PC_info:
    def __init__(self):
        """Provides some infomation:
        - hostname
        - platform
        - cpu
        - ram
        - gpu
        """
        self.hostname = platform.node()  # 获取计算机网络名称
        self.platform = platform.platform()  # 系统名称及版本号
        
        self.ram = str(round(psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0), 1))+'GB'
        
        _cpu_info = cpuinfo.get_cpu_info()
        self.cpu = '{} @{}GHz ({} cores)'.format(_cpu_info['brand_raw'], str(round(float(_cpu_info['hz_actual_friendly'].split()[0]), 1)), psutil.cpu_count(logical=False))
        
        # GPU Info
        pynvml.nvmlInit()
        self.gpu_driver_version = str(pynvml.nvmlSystemGetDriverVersion(), encoding="utf-8")  # 驱动版本号
        gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU数量
        self.num_gpu = str(gpuDeviceCount)
        _gpu_list = []
        for i in range(gpuDeviceCount):
            _handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            _gpu_model = str(pynvml.nvmlDeviceGetName(_handle), encoding="utf-8")  # 显卡型号
            
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(_handle)  # 显存信息
            _gpu_mem = str(round(meminfo.total / (1024.0 * 1024.0 * 1024.0)))+'GB'  # 总显存大小
            
            _gpu = '{} ({})'.format(_gpu_model, _gpu_mem)  # 显卡型号-显存
            
            _gpu_list.append(_gpu)
        pynvml.nvmlShutdown() #最后关闭管理工具
        
        self.gpu = ''
        for k, v in dict(Counter(_gpu_list)).items():
            self.gpu += '{} (x{}),'.format(k, v)
        self.gpu = self.gpu[:-1]


if __name__ == '__main__':
    pc = PC_info()
    
    print(pc.hostname)
    print(pc.platform)
    print(pc.cpu)
    print(pc.ram)
    print(pc.gpu)
