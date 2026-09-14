import os
import sys
import platform
import subprocess
import shutil
import json
import ctypes

def get_ram_info():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_ram_gb": round(stat.ullTotalPhys / (1024**3), 2),
        "avail_ram_gb": round(stat.ullAvailPhys / (1024**3), 2),
        "ram_load_pct": stat.dwMemoryLoad
    }

def find_tools():
    tools = ["git", "cmake", "ninja", "cl", "clang", "gcc", "g++", "vulkaninfo", "vkvia", "cargo", "rustc", "ollama", "ffmpeg"]
    found = {}
    for t in tools:
        p = shutil.which(t)
        found[t] = p
    return found

def get_vulkan_details():
    vk_info_path = shutil.which("vulkaninfo")
    vk_env = os.environ.get("VULKAN_SDK", None)
    details = {
        "vulkan_sdk_env": vk_env,
        "vulkaninfo_path": vk_info_path,
        "vulkan_installed": False,
        "devices": []
    }
    # Check vulkan-1.dll
    try:
        vk_dll = ctypes.windll.vulkan_1
        details["vulkan_1_dll"] = True
    except Exception as e:
        details["vulkan_1_dll"] = False
        details["vulkan_1_dll_error"] = str(e)

    if vk_info_path:
        try:
            res = subprocess.run([vk_info_path, "--summary"], capture_output=True, text=True, timeout=10)
            details["vulkaninfo_summary"] = res.stdout
            details["vulkan_installed"] = True
        except Exception as e:
            details["vulkaninfo_error"] = str(e)
    return details

def get_disk_info():
    usage = shutil.disk_usage("C:\\")
    return {
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }

def get_installed_pip_packages():
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True, timeout=10)
        return json.loads(res.stdout)
    except Exception as e:
        return {"error": str(e)}

def main():
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "ram": get_ram_info(),
        "disk": get_disk_info(),
        "tools": find_tools(),
        "vulkan": get_vulkan_details(),
        "pip_packages": get_installed_pip_packages()
    }
    print(json.dumps(info, indent=2))
    os.makedirs("diagnostics", exist_ok=True)
    with open("diagnostics/environment_report.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    main()
