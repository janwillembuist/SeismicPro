from functools import partial, wraps
from collections import defaultdict

from .utils import to_list
from ..batchflow import action, inbatch_parallel


def batch_method(*args, target="for", force=False):
    batch_method_params = {"target": target, "force": force}

    if len(args) == 1 and callable(args[0]):
        method = args[0]
        method.batch_method_params = batch_method_params
        return method

    if len(args) > 0:
        raise ValueError("batch_method decorator does not accept positional arguments")

    def decorator(method):
        method.batch_method_params = batch_method_params
        return method
    return decorator


def _apply_to_each_component(method, target, fetch_method_target):
    @wraps(method)
    def decorated_method(self, *args, src, dst=None, **kwargs):
        src_list = to_list(src)
        dst_list = to_list(dst) if dst is not None else src_list

        for src, dst in zip(src_list, dst_list):
            src_target = target  # set default target
            if fetch_method_target:  # dynamically fetch target from method attribute
                src_type_set = {type(elem) for elem in getattr(self, src)}
                if len(src_type_set) != 1:
                    err_msg = "Component elements must have the same type, but {} found"
                    raise ValueError(err_msg.format(", ".join([str(t) for t in src_type_set])))
                src_target = getattr(src_type_set.pop(), method.__name__).batch_method_params["target"]
            if "target" in kwargs:  # fetch target from passed kwargs
                src_target = kwargs.pop("target")
            parallel_method = inbatch_parallel(init="_init_component", target=src_target)(method)
            parallel_method(self, *args, src=src, dst=dst, **kwargs)
        return self
    return decorated_method


def apply_to_each_component(*args, target="for", fetch_method_target=True):
    partial_apply = partial(_apply_to_each_component, target=target, fetch_method_target=fetch_method_target)
    if len(args) == 1 and callable(args[0]):
        return partial_apply(args[0])
    return partial_apply


def get_class_methods(cls):
    return {func for func in dir(cls) if callable(getattr(cls, func))}


def create_batch_methods(*component_classes):
    def decorator(cls):
        decorated_methods = set()
        force_methods = set()
        for component_class in component_classes:
            for method_name in get_class_methods(component_class):
                method = getattr(component_class, method_name)
                if hasattr(method, "batch_method_params"):
                    decorated_methods.add(method_name)
                    if getattr(method, "batch_method_params")["force"]:
                        force_methods.add(method_name)
        methods_to_add = (decorated_methods - get_class_methods(cls)) | force_methods

        # TODO: dynamically generate docstring
        def create_method(method_name):
            def method(self, index, *args, src=None, dst=None, **kwargs):
                pos = self.index.get_pos(index)
                obj = getattr(self, src)[pos]
                if src != dst:
                    obj = obj.copy()
                getattr(self, dst)[pos] = getattr(obj, method_name)(*args, **kwargs)
            method.__name__ = method_name
            return action(apply_to_each_component(method))

        for method_name in methods_to_add:
            setattr(cls, method_name, create_method(method_name))
        return cls
    return decorator