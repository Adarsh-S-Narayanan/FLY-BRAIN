import os
import sys
sys.path.insert(0, os.path.abspath("."))
import struct
import numpy as np
import vulkan as vk
from src.compute.cpu_reference import cpu_brain_step

class VulkanBrainCompute:
    def __init__(self, spv_path: str = "shaders/brain_step.spv", enable_validation: bool = False):
        self.spv_path = spv_path
        self.enable_validation = enable_validation
        self.instance = None
        self.device = None
        self.queue = None
        self.queue_family_idx = None
        self.physical_device = None
        self.command_pool = None
        self.pipeline = None
        self.pipeline_layout = None
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.device_memory_properties = None
        self.device_name = "Unknown"
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
        self._init_pipeline()

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

    def _init_pipeline(self):
        with open(self.spv_path, "rb") as f:
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
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )

        pipe_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        self.pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipe_create_info], None)[0]

        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=9,
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
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
    ):
        N = len(potentials_in)

        # Prepare inputs as contiguous arrays
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

        # Create input buffers (bindings 0 to 5)
        for arr in arrays:
            b, m, size = self._create_buffer(arr.nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            ptr = vk.vkMapMemory(self.device, m, 0, arr.nbytes, 0)
            ptr[0:arr.nbytes] = arr.tobytes()
            vk.vkUnmapMemory(self.device, m)
            buffers.append(b)
            mems.append(m)
            buffer_sizes.append(arr.nbytes)

        # Create output buffers (bindings 6, 7): potentials_out, next_activations
        out_bytes = N * 4
        for _ in range(2):
            b, m, size = self._create_buffer(out_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buffers.append(b)
            mems.append(m)
            buffer_sizes.append(out_bytes)

        # Create params buffer (binding 8)
        p_buf, p_mem, p_size = self._create_buffer(len(params_bytes), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        ptr = vk.vkMapMemory(self.device, p_mem, 0, len(params_bytes), 0)
        ptr[0:len(params_bytes)] = params_bytes
        vk.vkUnmapMemory(self.device, p_mem)
        buffers.append(p_buf)
        mems.append(p_mem)
        buffer_sizes.append(len(params_bytes))

        # Allocate descriptor set
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        # Write descriptors for all 9 bindings
        writes = []
        buffer_infos = []
        for b_idx in range(9):
            b_info = vk.VkDescriptorBufferInfo(
                buffer=buffers[b_idx],
                offset=0,
                range=buffer_sizes[b_idx],
            )
            buffer_infos.append(b_info)
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

        # Allocate command buffer
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
        vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
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

        # Read back outputs
        ptr0 = vk.vkMapMemory(self.device, mems[6], 0, out_bytes, 0)
        potentials_out = np.frombuffer(ptr0, dtype=np.float32).copy()
        vk.vkUnmapMemory(self.device, mems[6])

        ptr1 = vk.vkMapMemory(self.device, mems[7], 0, out_bytes, 0)
        next_activations = np.frombuffer(ptr1, dtype=np.float32).copy()
        vk.vkUnmapMemory(self.device, mems[7])

        # Cleanup per-step resources
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cmd_buf])
        vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        for b in buffers:
            vk.vkDestroyBuffer(self.device, b, None)
        for m in mems:
            vk.vkFreeMemory(self.device, m, None)

        return potentials_out, next_activations

    def cleanup(self):
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
            if self.pipeline:
                vk.vkDestroyPipeline(self.device, self.pipeline, None)
            if self.pipeline_layout:
                vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
            if self.descriptor_pool:
                vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            if self.descriptor_set_layout:
                vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
            if self.command_pool:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None

if __name__ == "__main__":
    from src.connectome.loader import get_or_create_circuit
    circuit = get_or_create_circuit(512, cache_name="test_circuit_512.npz")
    print(f"Loaded circuit: {circuit.num_neurons} neurons, {circuit.num_synapses} synapses")
    
    rng = np.random.RandomState(42)
    prev_act = rng.uniform(0.0, 1.0, circuit.num_neurons).astype(np.float32)
    ext_in = rng.uniform(0.0, 0.5, circuit.num_neurons).astype(np.float32)
    pot_in = rng.uniform(-0.2, 0.2, circuit.num_neurons).astype(np.float32)

    print("Running CPU reference step...")
    cpu_pot, cpu_act = cpu_brain_step(
        circuit.row_offsets,
        circuit.col_indices,
        circuit.weights,
        prev_act,
        ext_in,
        pot_in
    )
    
    print("Running Vulkan compute step...")
    vk_engine = VulkanBrainCompute()
    print(f"Using GPU Device: {vk_engine.device_name}")
    gpu_pot, gpu_act = vk_engine.run_step(
        circuit.row_offsets,
        circuit.col_indices,
        circuit.weights,
        prev_act,
        ext_in,
        pot_in
    )
    vk_engine.cleanup()

    diff_pot = float(np.max(np.abs(cpu_pot - gpu_pot)))
    diff_act = float(np.max(np.abs(cpu_act - gpu_act)))
    print(f"Max Potential Difference (CPU vs Vulkan GPU): {diff_pot:.6e}")
    print(f"Max Activation Difference (CPU vs Vulkan GPU): {diff_act:.6e}")
    assert diff_pot < 1e-4, f"Potential difference too large: {diff_pot}"
    assert diff_act < 1e-4, f"Activation difference too large: {diff_act}"
    print("SUCCESS: Vulkan GPU output matches CPU reference within numerical tolerance!")
