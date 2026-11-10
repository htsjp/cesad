from phase_xa.provenance import provenance_aligned


def assert_cache_alignment(image_cache, patch_cache):
    image_ok = provenance_aligned(image_cache["keys"], image_cache["values"], image_cache["provenance"])
    patch_ok = provenance_aligned(patch_cache["keys"], patch_cache["values"], patch_cache["provenance"])
    if not image_ok or not patch_ok:
        raise RuntimeError("cache key/value/provenance alignment failed")
    return {"image_cache_aligned": image_ok, "patch_cache_aligned": patch_ok}
