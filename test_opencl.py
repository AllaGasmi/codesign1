import pyopencl as cl
platforms = cl.get_platforms()
for platform in platforms:
    print(f"Platform: {platform.name}")
    devices = platform.get_devices()
    for device in devices:
        print(f"\n  Device: {device.name}")
        print(f"  Global Memory  : {device.global_mem_size // (1024*1024)} MB")
        print(f"  Local Memory   : {device.local_mem_size // 1024} KB")
        print(f"  Cache Memory   : {device.global_mem_cache_size // 1024} KB")
        print(f"  Compute Units  : {device.max_compute_units}")
        print(f"  Max Work-group : {device.max_work_group_size}")
        print(f"  Max Work-items : {device.max_work_item_sizes}")