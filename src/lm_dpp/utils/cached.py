import pickle
from pathlib import Path


class cached_ddict(dict):
    """A defaultdict-like dictionary that saves itself to disk after any modification."""

    def __init__(self, path, read_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cow = False
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            with open(self.path, "rb") as f:
                prev = pickle.load(f)
                if isinstance(prev, dict):
                    self.update(prev)
                for k, v in self.items():
                    if isinstance(v, dict):
                        self[k] = parented_dict(self, dict(v))
        else:
            print(f"Creating new cache at {self.path}")
        self.cow = not read_only

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump({k: dict(v) if isinstance(v, parented_dict) else v for k, v in self.items()}, f)

    def __getitem__(self, key):
        if key not in self:
            self[key] = parented_dict(self)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = parented_dict(self, value)
        super().__setitem__(key, value)
        self._save() if self.cow else None


class parented_dict(dict):
    """A dictionary that notifies its parent dictionary when modified."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._parent._save() if self._parent.cow else None

    def __delitem__(self, key):
        super().__delitem__(key)
        self._parent._save() if self._parent.cow else None


class cached_list(list):
    """A list that automatically saves to disk after modifications."""

    def __init__(self, path, *args):
        super().__init__(*args)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            with open(self.path, "rb") as f:
                prev = pickle.load(f)
                if isinstance(prev, list):
                    self.extend(prev)
        else:
            pass  # new file, created silently

        self._save()

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(list(self), f)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._save()

    def append(self, value):
        super().append(value)
        self._save()

    def extend(self, iterable):
        super().extend(iterable)
        self._save()

    def insert(self, index, value):
        super().insert(index, value)
        self._save()

    def pop(self, index=-1):
        value = super().pop(index)
        self._save()
        return value

    def remove(self, value):
        super().remove(value)
        self._save()

    def clear(self):
        super().clear()
        self._save()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs)
        self._save()

    def reverse(self):
        super().reverse()
        self._save()
