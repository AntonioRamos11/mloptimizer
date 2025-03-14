import GPUtil
gpus=  GPUtil.getGPUs()
try:
    gpu_info = [{'id': gpu.id, 'name': gpu.name, 'load': gpu.load, 
            'memoryTotal': gpu.memoryTotal, 'memoryUsed': gpu.memoryUsed, 
            'temperature': gpu.temperature} for gpu in gpus]
    print(gpu_info)
except Exception as e:
    print(f"Error collecting GPU information: {e}")