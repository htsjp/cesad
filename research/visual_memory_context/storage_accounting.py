def adapter_storage(param_count):
    return {"adapter_parameters": int(param_count), "adapter_fp32_bytes": int(param_count) * 4, "adapter_fp16_bytes": int(param_count) * 2}

