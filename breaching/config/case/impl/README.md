### Dataloader implementation choices:
#### Turn these off for better reproducibility of experiments
- shuffle: @boolean DEFAULT=FALSE
  - Samples elements randomly. If without replacement, then sample from a shuffled dataset.
- sample_with_replacement: @boolean DEFAULT=False
  - Used along with shuffle. Samples are drawn on-demand with replacement if True

### PyTorch configuration
- dtype: @torch.dtype DEFAULT=float
  - This has to be float when mixed_precision is True. A torch.dtype is an object that represents the data type of torch.Tensor. PyTorch has several different data types, refer to https://docs.pytorch.org/docs/stable/tensor_attributes.html for more info.

- non_blocking: @boolean DEFAULT=True
  - Not used.

- sharing_strategy: @string DEFAULT=file_descriptor
  - Defines the strategy of torch.multiprocessing to provide shared views on the same data in diferent processes. Refer to https://docs.pytorch.org/docs/stable/multiprocessing.html#sharing-strategies for more info.

- enable_gpu_acc: @boolean DEFAULT=FALSE
  - Uses CUDA as torch device instead of CPU. 

- benchmark: @boolean DEFAULT=True
  - Causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
- deterministic: @boolean DEFAULT=False 
  - This option will disable cuDNN non-deterministic ops.

- pin_memory: @boolean DEFAULT=True
- threads: @int DEFAULT=0
  - Maximal number of cpu dataloader workers used per GPU
- persistent_workers: @boolean DEFAULT=False

- mixed_precision: @boolean DEFAULT=False
- grad_scaling: @boolean DEFAULT=True 
  - This is a no-op if mixed-precision is off
- JIT: "script"|"trace"|null DEFAULT=null 
  - script currently break autocast mixed precision
  - trace breaks training

- validate_every_nth_step: @int DEFAULT=10

- checkpoint: @object 
	checkpont.name: @string
  - checkpoint.save_every_nth_step: @int DEFAULT=10

- enable_huggingface_offline_mode: @boolean DEFAULT=True 
  - huggingface` needs an internet connection for metrics, datasets and tokenizers. After caching these objects, it can be turned to offline mode with this argument.
