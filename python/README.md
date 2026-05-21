# axon-format

Python bindings and pure-Python fallback reader for Axon, the Adaptive
eXecutable Object Notation model-weight container.

```python
import axon

model = axon.load("model.axon")
print(model.summary())
tensor = model["emb_weight"]
```

The package uses the native `axon_ffi` library when it can find one and falls
back to the pure-Python reader for basic loading workflows.
