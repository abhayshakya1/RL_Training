from verifiers.v1.harnesses.null import NullHarness

from geo_distance.taskset import GeoDistanceTaskset

# NullHarness is exported so this taskset's default harness is a single plain model
# turn. Without a Harness in __all__ the loader falls back to "bash", which both
# provisions a container and hands the model a shell it could compute the answer in.
__all__ = ["GeoDistanceTaskset", "NullHarness"]
