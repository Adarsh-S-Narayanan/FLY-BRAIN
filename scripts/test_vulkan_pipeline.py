import struct
import numpy as np
import vulkan as vk

def test_vulkan_pipeline():
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="FlyBrainVulkan",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="FlyBrainEngine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
    )
    
    instance = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            enabledLayerCount=0,
            ppEnabledLayerNames=[],
        ),
        None
    )
    
    physical_device = vk.vkEnumeratePhysicalDevices(instance)[0]
    props = vk.vkGetPhysicalDeviceProperties(physical_device)
    print(f"Selected device: {props.deviceName}")
    
    # Find compute queue family
    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    compute_family_idx = None
    for idx, qf in enumerate(queue_families):
        if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
            compute_family_idx = idx
            break
            
    print(f"Compute queue family: {compute_family_idx}")
    
    # Create device
    queue_create_info = vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=compute_family_idx,
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    
    device = vk.vkCreateDevice(
        physical_device,
        vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            pEnabledFeatures=None,
        ),
        None
    )
    print("Vulkan logical device created successfully:", device)
    
    queue = vk.vkGetDeviceQueue(device, compute_family_idx, 0)
    print("Compute queue acquired:", queue)
    
    vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)
    print("Clean teardown successful.")

if __name__ == "__main__":
    test_vulkan_pipeline()
