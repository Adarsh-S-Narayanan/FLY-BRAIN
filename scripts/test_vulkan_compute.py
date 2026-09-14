import sys
import os
import vulkan as vk

def main():
    print("Testing Vulkan compute setup...")
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="FlyBrainVulkan",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="FlyBrainEngine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
    )
    
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=[],
        enabledLayerCount=0,
        ppEnabledLayerNames=[],
    )
    
    instance = vk.vkCreateInstance(create_info, None)
    print("Vulkan Instance created successfully:", instance)
    
    devices = vk.vkEnumeratePhysicalDevices(instance)
    print(f"Found {len(devices)} physical device(s).")
    
    for i, dev in enumerate(devices):
        props = vk.vkGetPhysicalDeviceProperties(dev)
        print(f"Device {i}: {props.deviceName} (Type: {props.deviceType}, Driver: {props.driverVersion})")
    
    vk.vkDestroyInstance(instance, None)
    print("Vulkan test passed cleanly!")

if __name__ == "__main__":
    main()
