import os
import sys
import struct
import numpy as np
import vulkan as vk
from typing import Tuple, Optional

class VulkanComputeEngine:
    """
    Production Vulkan 1.2+ compute engine executing neural activation and plasticity
    shaders directly on physical GPU devices (AMD Radeon Graphics).
    """
    def __init__(
        self,
        brain_spv_path: str = "shaders/brain_step.spv",
        plasticity_spv_path: str = "shaders/plasticity.spv",
        enable_validation: bool = False
    ):
        self.brain_spv_path = brain_spv_path
        self.plasticity_spv_path = plasticity_spv_path
        self.enable_validation = enable_validation
        
        self.instance = None
        self.device = None
        self.queue = None
        self.queue_family_idx = None
        self.physical_device = None
        self.command_pool = None
        self.device_memory_properties = None
        self.device_name = "Unknown"
        self.device_type = 0
        self.driver_version = 0
        
        # Pipelines
        self.brain_pipeline = None
        self.brain_layout = None
        self.brain_desc_layout = None
        
        self.plasticity_pipeline = None
        self.plasticity_layout = None
        self.plasticity_desc_layout = None
        
        self.descriptor_pool = None
        self._init_vulkan()

    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        for i in range(self.device_memory_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (self.device_memory_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type!")

    def _init_vulkan(self):
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="FlyBrainVulkanEngine",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="FlyBrain",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
        )

        layers = []
        if self.enable_validation:
            layers.append("VK_LAYER_KHRONOS_validation")

        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

        pdevs = vk.vkEnumeratePhysicalDevices(self.instance)
        if not pdevs:
            raise RuntimeError("No Vulkan physical devices found!")
        self.physical_device = pdevs[0]
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = props.deviceName
        self.device_type = props.deviceType
        self.driver_version = props.driverVersion
        self.device_memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)

        q_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        for idx, qf in enumerate(q_props):
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.queue_family_idx = idx
                break
        if self.queue_family_idx is None:
            raise RuntimeError("No compute queue family available!")

        q_create = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family_idx,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        d_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[q_create],
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            pEnabledFeatures=None,
        )
        self.device = vk.vkCreateDevice(self.physical_device, d_create, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family_idx, 0)

        cmd_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_idx,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
        
        # Build pipelines
        self._init_brain_pipeline()
        self._init_plasticity_pipeline()
        
        # Build descriptor pool
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1024,
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=128,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

    def _create_buffer(self, size_bytes: int, usage: int):
        buf_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self.device, buf_info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self.device, buf)
        mem_type_idx = self._find_memory_type(
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type_idx,
        )
        mem = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        return buf, mem, reqs.size

    def _init_brain_pipeline(self):
        if not os.path.exists(self.brain_spv_path):
            raise FileNotFoundError(f"Shader not found: {self.brain_spv_path}")
        with open(self.brain_spv_path, "rb") as f:
            code = f.read()

        mod_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        shader_module = vk.vkCreateShaderModule(self.device, mod_info, None)

        bindings = []
        for b in range(9):
            bindings.append(
                vk.VkDescriptorSetLayoutBinding(
                    binding=b,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers=None,
                )
            )

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.brain_desc_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.brain_desc_layout],
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        self.brain_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )

        pipe_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.brain_layout,
        )
        self.brain_pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipe_create_info], None)[0]
        vk.vkDestroyShaderModule(self.device, shader_module, None)

    def _init_plasticity_pipeline(self):
        if not os.path.exists(self.plasticity_spv_path):
            raise FileNotFoundError(f"Shader not found: {self.plasticity_spv_path}")
        with open(self.plasticity_spv_path, "rb") as f:
            code = f.read()

        mod_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        shader_module = vk.vkCreateShaderModule(self.device, mod_info, None)

        bindings = []
        for b in range(6):
            bindings.append(
                vk.VkDescriptorSetLayoutBinding(
                    binding=b,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers=None,
                )
            )

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.plasticity_desc_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.plasticity_desc_layout],
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        self.plasticity_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )

        pipe_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.plasticity_layout,
        )
        self.plasticity_pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipe_create_info], None)[0]
        vk.vkDestroyShaderModule(self.device, shader_module, None)

    def run_step(
        self,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        weights: np.ndarray,
        prev_activations: np.ndarray,
        external_inputs: np.ndarray,
        potentials_in: np.ndarray,
        decay: float = 0.85,
        threshold: float = 0.5,
        leak: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(potentials_in)

        row_offsets = np.ascontiguousarray(row_offsets, dtype=np.int32)
        col_indices = np.ascontiguousarray(col_indices, dtype=np.int32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        prev_activations = np.ascontiguousarray(prev_activations, dtype=np.float32)
        external_inputs = np.ascontiguousarray(external_inputs, dtype=np.float32)
        potentials_in = np.ascontiguousarray(potentials_in, dtype=np.float32)
        params_bytes = struct.pack('ifff', N, decay, threshold, leak)

        arrays = [
            row_offsets, col_indices, weights, prev_activations,
            external_inputs, potentials_in
        ]
        buffers = []
        mems = []
        buffer_sizes = []

        for arr in arrays:
            b, m, _ = self._create_buffer(arr.nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            ptr = vk.vkMapMemory(self.device, m, 0, arr.nbytes, 0)
            ptr[0:arr.nbytes] = arr.tobytes()
            vk.vkUnmapMemory(self.device, m)
            buffers.append(b)
            mems.append(m)
            buffer_sizes.append(arr.nbytes)

        out_bytes = N * 4
        for _ in range(2):
            b, m, _ = self._create_buffer(out_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buffers.append(b)
            mems.append(m)
            buffer_sizes.append(out_bytes)

        p_buf, p_mem, _ = self._create_buffer(len(params_bytes), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        ptr = vk.vkMapMemory(self.device, p_mem, 0, len(params_bytes), 0)
        ptr[0:len(params_bytes)] = params_bytes
        vk.vkUnmapMemory(self.device, p_mem)
        buffers.append(p_buf)
        mems.append(p_mem)
        buffer_sizes.append(len(params_bytes))

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.brain_desc_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        writes = []
        for b_idx in range(9):
            b_info = vk.VkDescriptorBufferInfo(
                buffer=buffers[b_idx],
                offset=0,
                range=buffer_sizes[b_idx],
            )
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=b_idx,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[b_info],
                )
            )
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        cmd_alloc = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd_buf = vk.vkAllocateCommandBuffers(self.device, cmd_alloc)[0]

        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd_buf, begin_info)
        vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.brain_pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.brain_layout,
            0,
            1,
            [descriptor_set],
            0,
            None
        )

        group_size = 64
        num_groups = (N + group_size - 1) // group_size
        vk.vkCmdDispatch(cmd_buf, num_groups, 1, 1)
        vk.vkEndCommandBuffer(cmd_buf)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd_buf],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        ptr0 = vk.vkMapMemory(self.device, mems[6], 0, out_bytes, 0)
        potentials_out = np.frombuffer(ptr0, dtype=np.float32).copy()
        vk.vkUnmapMemory(self.device, mems[6])

        ptr1 = vk.vkMapMemory(self.device, mems[7], 0, out_bytes, 0)
        next_activations = np.frombuffer(ptr1, dtype=np.float32).copy()
        vk.vkUnmapMemory(self.device, mems[7])

        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cmd_buf])
        vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        for b in buffers:
            vk.vkDestroyBuffer(self.device, b, None)
        for m in mems:
            vk.vkFreeMemory(self.device, m, None)

        return potentials_out, next_activations

    def run_plasticity_step(
        self,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        weights: np.ndarray,
        post_activations: np.ndarray,
        pre_activations: np.ndarray,
        learning_rate: float,
        reward: float,
        weight_decay: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 1.0
    ) -> np.ndarray:
        M = len(weights)
        row_offsets = np.ascontiguousarray(row_offsets, dtype=np.int32)
        col_indices = np.ascontiguousarray(col_indices, dtype=np.int32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        post_activations = np.ascontiguousarray(post_activations, dtype=np.float32)
        pre_activations = np.ascontiguousarray(pre_activations, dtype=np.float32)
        params_bytes = struct.pack('ifffff', M, learning_rate, reward, weight_decay, min_weight, max_weight)

        arrays = [row_offsets, col_indices, weights, post_activations, pre_activations]
        buffers = []
        mems = []
        buffer_sizes = []

        for arr in arrays:
            b, m, _ = self._create_buffer(arr.nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            ptr = vk.vkMapMemory(self.device, m, 0, arr.nbytes, 0)
            ptr[0:arr.nbytes] = arr.tobytes()
            vk.vkUnmapMemory(self.device, m)
            buffers.append(b)
            mems.append(m)
            buffer_sizes.append(arr.nbytes)

        p_buf, p_mem, _ = self._create_buffer(len(params_bytes), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        ptr = vk.vkMapMemory(self.device, p_mem, 0, len(params_bytes), 0)
        ptr[0:len(params_bytes)] = params_bytes
        vk.vkUnmapMemory(self.device, p_mem)
        buffers.append(p_buf)
        mems.append(p_mem)
        buffer_sizes.append(len(params_bytes))

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.plasticity_desc_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        writes = []
        for b_idx in range(6):
            b_info = vk.VkDescriptorBufferInfo(
                buffer=buffers[b_idx],
                offset=0,
                range=buffer_sizes[b_idx],
            )
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=b_idx,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[b_info],
                )
            )
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        cmd_alloc = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd_buf = vk.vkAllocateCommandBuffers(self.device, cmd_alloc)[0]

        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd_buf, begin_info)
        vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.plasticity_pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.plasticity_layout,
            0,
            1,
            [descriptor_set],
            0,
            None
        )

        group_size = 64
        num_groups = (M + group_size - 1) // group_size
        vk.vkCmdDispatch(cmd_buf, num_groups, 1, 1)
        vk.vkEndCommandBuffer(cmd_buf)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd_buf],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        ptr_w = vk.vkMapMemory(self.device, mems[2], 0, weights.nbytes, 0)
        updated_weights = np.frombuffer(ptr_w, dtype=np.float32).copy()
        vk.vkUnmapMemory(self.device, mems[2])

        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cmd_buf])
        vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        for b in buffers:
            vk.vkDestroyBuffer(self.device, b, None)
        for m in mems:
            vk.vkFreeMemory(self.device, m, None)

        return updated_weights

    def cleanup(self):
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
            if self.brain_pipeline:
                vk.vkDestroyPipeline(self.device, self.brain_pipeline, None)
            if self.brain_layout:
                vk.vkDestroyPipelineLayout(self.device, self.brain_layout, None)
            if self.brain_desc_layout:
                vk.vkDestroyDescriptorSetLayout(self.device, self.brain_desc_layout, None)
            if self.plasticity_pipeline:
                vk.vkDestroyPipeline(self.device, self.plasticity_pipeline, None)
            if self.plasticity_layout:
                vk.vkDestroyPipelineLayout(self.device, self.plasticity_layout, None)
            if self.plasticity_desc_layout:
                vk.vkDestroyDescriptorSetLayout(self.device, self.plasticity_desc_layout, None)
            if self.descriptor_pool:
                vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            if self.command_pool:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None
