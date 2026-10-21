import torch
import comfy.model_management
from ..core import logger
import os
import platform

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson device by checking the device model
    information or the platform release.
    """
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception as e:
        # logger.warning(f"JETSON: Could not read /proc/device-tree/model: {e} (If you're not using Jetson, ignore this warning)")
        # If /proc/device-tree/model is not available, check platform.release()
        platform_release = platform.release()
        logger.info(f"Platform release: {platform_release}")
        if 'tegra' in platform_release.lower():
            logger.info("Detected 'tegra' in platform release. Assuming Jetson device.")
            return True
        else:
            logger.info("JETSON: Not detected.")
            return False

IS_JETSON = is_jetson()

class CGPUInfo:
    """
    This class is responsible for getting information from GPU (ONLY).
    """
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []
    nvmlDevices = []

    def __init__(self):
        if IS_JETSON:
            # Try to import jtop for Jetson devices
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('jtop initialized on Jetson device.')
            except ImportError as e:
                logger.error('jtop is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not initialize jtop. ' + str(e))
        else:
            # Try to import pynvml for non-Jetson devices
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.nvmlDevices = self.getNvmlDevices()
                self.pynvmlLoaded = True
                logger.info('pynvml (NVIDIA) initialized.')
            except ImportError as e:
                logger.error('pynvml is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not init pynvml (NVIDIA). ' + str(e))

        self.anygpuLoaded = self.pynvmlLoaded or self.jtopLoaded

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Could not pick default device. ' + str(e))

        if self.pynvmlLoaded and not self.jtopLoaded and not self.deviceGetCount():
            logger.warning('No GPU detected, disabling GPU monitoring.')
            self.anygpuLoaded = False
            self.pynvmlLoaded = False
            self.jtopLoaded = False

        if self.anygpuLoaded:
            if self.deviceGetCount() > 0:
                self.cudaDevicesFound = self.deviceGetCount()

                logger.info(f"GPU/s:")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"{deviceIndex}) {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    # Same index as gpus, with default values
                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                logger.warning('No GPU with CUDA detected.')
        else:
            logger.warning('No GPU monitoring libraries available.')

        self.cudaDevice = 'cpu' if self.torchDevice == 'cpu' else 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

        if self.cuda and self.cudaAvailable and self.torchDevice == 'cpu':
            logger.warning('CUDA is available, but torch is using CPU.')

    def getInfo(self):
        logger.debug('Getting GPUs info...')
        return self.gpus

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ''
        gpus = []

        if self.cudaDevice == 'cpu':
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        else:
            gpuType = self.cudaDevice

            if self.anygpuLoaded and self.cuda and self.cudaAvailable:
                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    gpuTemperature = -1

                    # GPU Utilization
                    if self.switchGPU and self.gpusUtilization[deviceIndex]:
                        try:
                            gpuUtilization = self.deviceGetUtilizationRates(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU utilization. ' + str(e))
                            logger.error('Monitor of GPU is turning off.')
                            self.switchGPU = False

                    if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                        try:
                            memory = self.deviceGetMemoryInfo(deviceHandle)
                            vramUsed = memory['used']
                            vramTotal = memory['total']

                            # Check if vramTotal is not zero or None
                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.error('Could not get GPU memory info. ' + str(e))
                            self.switchVRAM = False

                    # Temperature
                    if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                        try:
                            gpuTemperature = self.deviceGetTemperature(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU temperature. Turning off this feature. ' + str(e))
                            self.switchTemperature = False

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                    })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def getNvmlDevices(self):
        """
        Lists the devices to monitor. A GPU in MIG mode is replaced by its visible MIG devices,
        because the parent GPU does not report memory or utilization in MIG mode.
        """
        devices = []
        for index in range(self.pynvml.nvmlDeviceGetCount()):
            parent = self.pynvml.nvmlDeviceGetHandleByIndex(index)

            migDevices = []
            try:
                currentMode, _ = self.pynvml.nvmlDeviceGetMigMode(parent)
                if currentMode == self.pynvml.NVML_DEVICE_MIG_ENABLE:
                    for migIndex in range(self.pynvml.nvmlDeviceGetMaxMigDeviceCount(parent)):
                        try:
                            mig = self.pynvml.nvmlDeviceGetMigDeviceHandleByIndex(parent, migIndex)
                        except self.pynvml.NVMLError:
                            continue  # slot empty or not visible in this container
                        migDevices.append({
                            'handle': mig,
                            'parent': parent,
                            'gpuInstanceId': self.pynvml.nvmlDeviceGetGpuInstanceId(mig),
                            'gpmSample': None,
                        })
            except self.pynvml.NVMLError:
                pass  # MIG not supported on this GPU

            if migDevices:
                logger.info(f"GPU {index} is in MIG mode with {len(migDevices)} visible MIG device(s).")
                devices.extend(migDevices)
            else:
                devices.append({'handle': parent, 'parent': parent, 'gpuInstanceId': None, 'gpmSample': None})

        return devices

    def deviceGetCount(self):
        if self.pynvmlLoaded:
            return len(self.nvmlDevices)
        elif self.jtopLoaded:
            # For Jetson devices, we assume there's one GPU
            return 1
        else:
            return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded:
            return self.nvmlDevices[index]
        elif self.jtopLoaded:
            return index  # On Jetson, index acts as handle
        else:
            return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.pynvmlLoaded:
            gpuName = 'Unknown GPU'

            try:
                gpuName = self.pynvml.nvmlDeviceGetName(deviceHandle['handle'])
                try:
                    gpuName = gpuName.decode('utf-8', errors='ignore')
                except AttributeError:
                    pass

            except UnicodeDecodeError as e:
                gpuName = 'Unknown GPU (decoding error)'
                logger.error(f"UnicodeDecodeError: {e}")

            return gpuName
        elif self.jtopLoaded:
            # Access the GPU name from self.jtopInstance.gpu
            try:
                gpu_info = self.jtopInstance.gpu
                gpu_name = next(iter(gpu_info.keys()))
                return gpu_name
            except Exception as e:
                logger.error('Could not get GPU name. ' + str(e))
                return 'Unknown GPU'
        else:
            return ''

    def systemGetDriverVersion(self):
        if self.pynvmlLoaded:
            return f'NVIDIA Driver: {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.jtopLoaded:
            # No direct method to get driver version from jtop
            return 'NVIDIA Driver: unknown'
        else:
            return 'Driver unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        if self.pynvmlLoaded:
            if deviceHandle['gpuInstanceId'] is None:
                return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle['handle']).gpu
            return self.migGetUtilization(deviceHandle)
        elif self.jtopLoaded:
            # GPU utilization from jtop stats
            try:
                gpu_util = self.jtopInstance.stats.get('GPU', -1)
                return gpu_util
            except Exception as e:
                logger.error('Could not get GPU utilization. ' + str(e))
                return -1
        else:
            return 0

    def migGetUtilization(self, device):
        """
        nvmlDeviceGetUtilizationRates is not supported on MIG, so use the GPM SM utilization
        metric between the previous poll's sample and this one. The first poll returns 0.
        """
        sample = self.pynvml.nvmlGpmSampleAlloc()
        try:
            self.pynvml.nvmlGpmMigSampleGet(device['parent'], device['gpuInstanceId'], sample)
        except Exception:
            self.pynvml.nvmlGpmSampleFree(sample)
            raise

        previous = device['gpmSample']
        device['gpmSample'] = sample
        if previous is None:
            return 0

        try:
            metrics = self.pynvml.c_nvmlGpmMetricsGet_t()
            metrics.version = self.pynvml.NVML_GPM_METRICS_GET_VERSION
            metrics.numMetrics = 1
            metrics.sample1 = previous
            metrics.sample2 = sample
            metrics.metrics[0].metricId = self.pynvml.NVML_GPM_METRIC_SM_UTIL
            self.pynvml.nvmlGpmMetricsGet(metrics)
            if metrics.metrics[0].nvmlReturn != self.pynvml.NVML_SUCCESS:
                raise self.pynvml.NVMLError(metrics.metrics[0].nvmlReturn)
            return round(metrics.metrics[0].value)
        finally:
            self.pynvml.nvmlGpmSampleFree(previous)

    def deviceGetMemoryInfo(self, deviceHandle):
        if self.pynvmlLoaded:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle['handle'])
            return {'total': mem.total, 'used': mem.used}
        elif self.jtopLoaded:
            mem_data = self.jtopInstance.memory['RAM']
            total = mem_data['tot']
            used = mem_data['used']
            return {'total': total, 'used': used}
        else:
            return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        if self.pynvmlLoaded:
            # MIG devices have no sensor of their own; report the parent GPU's temperature.
            return self.pynvml.nvmlDeviceGetTemperature(deviceHandle['parent'], self.pynvml.NVML_TEMPERATURE_GPU)
        elif self.jtopLoaded:
            try:
                temperature = self.jtopInstance.stats.get('Temp gpu', -1)
                return temperature
            except Exception as e:
                logger.error('Could not get GPU temperature. ' + str(e))
                return -1
        else:
            return 0

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()
