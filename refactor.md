# Refactoring Approach for Parallel Candidate Optimization

## Current Architecture Problem

### CURRENT: Sequential Trial Loop
Each trial is completely independent—no cross-candidate communication during optimization.

```python
for trial in range(num_trials):
    candidate = _initialize_data()  # isolated
    for iter in range(max_iterations):
        optimize(candidate)         # isolated
    solutions.append(candidate)
select_best(solutions)
```

## Target Architecture

### PROPOSED: Parallel Candidate Optimization
All candidates exist simultaneously, allowing for consensus computation.

```python
candidates = [_initialize_data(seed=i) for i in range(trials)]
for iter in range(max_iterations):
    consensus = compute_consensus(candidates)   # shared info
    for candidate in candidates:
        loss = gradient_loss + λ * ||candidate - consensus||²
        loss.backward()
    optimizer.step()
select_best(candidates)
```

## Implementation Strategy

### Option A: List-Based Approach (Easier, more memory)
Maintain candidates as a Python list, share the optimization loop.

```python
def reconstruct(self, server_payload, shared_data, ...):
    rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
    
    # 1. Initialize ALL candidates at once with different seeds
    num_trials = self.cfg.restarts.num_trials
    candidates = []
    for trial in range(num_trials):
        torch.manual_seed(self.cfg.seed + trial)  # Different seed per trial
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        candidates.append(candidate)
    
    # 2. Setup GroupRegularization with reference to candidate list
    group_reg = self._find_group_regularizer()
    if group_reg is not None:
        group_reg.x_list = candidates  # Pass reference to regularizer
    
    # 3. Single joint optimization loop
    best_candidates = self._run_joint_optimization(rec_models, shared_data, labels, 
                                                    candidates, stats)
    
    # 4. Score and select best
    scores = torch.zeros(num_trials)
    for trial, candidate in enumerate(best_candidates):
        scores[trial] = self._score_trial(candidate, labels, rec_models, shared_data)
    
    optimal_solution = self._select_optimal_reconstruction(best_candidates, scores, stats)
    return dict(data=optimal_solution, labels=labels), stats
```

### Option B: Batched Tensor Approach (More efficient, complex)
Stack all candidates into a single tensor with an extra batch dimension.

```python
# Shape: [num_trials, num_data_points, C, H, W]
candidates = torch.stack([
    self._initialize_data(..., seed=trial) 
    for trial in range(num_trials)
], dim=0)

# Optimizer operates on the batched tensor
optimizer = torch.optim.Adam([candidates], lr=...)
```

## Key Refactoring Points

### 1. New `_run_joint_optimization` Method
Replace `_run_trial` with a joint version:

```python
def _run_joint_optimization(self, rec_model, shared_data, labels, candidates, stats):
    """Run optimization for all candidates simultaneously."""
    
    # Initialize regularizers
    for regularizer in self.regularizers:
        regularizer.initialize(rec_model, shared_data, labels)
    self.objective.initialize(self.loss_fn, self.cfg.impl, 
                              shared_data[0]["metadata"]["local_hyperparams"])
    
    # Track best per candidate
    best_candidates = [c.detach().clone() for c in candidates]
    minimal_values = [torch.tensor(float("inf"), **self.setup) for _ in candidates]
    
    # Single optimizer for ALL candidates
    optimizer, scheduler = self._init_optimizer(candidates)  # Modify to accept list
    
    for iteration in range(self.cfg.optim.max_iterations):
        optimizer.zero_grad()
        
        total_objective = 0
        for idx, candidate in enumerate(candidates):
            # Compute individual objective
            obj = self._compute_single_objective(candidate, labels, rec_model, 
                                                  shared_data, iteration)
            total_objective += obj
            
            # Track best
            with torch.no_grad():
                if obj < minimal_values[idx]:
                    minimal_values[idx] = obj.detach()
                    best_candidates[idx] = candidate.detach().clone()
        
        # Backward and step (regularizers like GroupRegularization 
        # automatically access self.x_list during forward())
        total_objective.backward()
        optimizer.step()
        scheduler.step()
        
        # Boxed projection for all
        with torch.no_grad():
            if self.cfg.optim.boxed:
                for candidate in candidates:
                    candidate.data = torch.clamp(candidate, 
                                                  -self.dm / self.ds, 
                                                  (1 - self.dm) / self.ds)
    
    return best_candidates
```

### 2. Modify `_init_optimizer` to Accept Multiple Candidates

```python
def _init_optimizer(self, candidates):
    """Accept list of candidates instead of single candidate."""
    if isinstance(candidates, list):
        params = candidates  # All are nn.Parameter-like
    else:
        params = [candidates]
    
    optimizer, scheduler = optimizer_lookup(
        params,
        self.cfg.optim.optimizer,
        self.cfg.optim.step_size,
        ...
    )
    return optimizer, scheduler
```

### 3. Updated `GroupRegularization` Integration
The existing `GroupRegularization` class is already designed well! It:
- Computes consensus via `compute_consensus(x_list)`
- Returns `||candidate - x_C||²` loss scaled by `self.scale`

The key is to **pass the live candidate list** before each iteration:

```python
# During optimization, before calling forward on regularizers:
if hasattr(group_reg, 'x_list'):
    group_reg.x_list = [c.detach() for c in candidates]  # No grad through others
```

## Memory and Performance Considerations

| Factor | Sequential (Current) | Parallel (Proposed) |
|--------|---------------------|---------------------|
| **Memory** | `O(1)` candidates at once | `O(num_trials)` candidates |
| **GPU Util** | Low (single sample batches) | High (can batch forward passes) |
| **Consensus** | Not possible | ✅ Computable each iteration |
| **Time** | `num_trials × iterations` | `iterations` (with batching) |

For **8 trials** with **224×224 RGB images**, additional memory ≈ `8 × 3 × 224 × 224 × 4 bytes ≈ 4.8 MB` per batch—negligible on modern GPUs.

## Recommended Files to Modify

1. **[`optimization_based_attack.py`](file:///home/siladittyamanna/Documents/smanna/iisc/work1/breaching/breaching/attacks/optimization_based_attack.py)**
   - Replace `reconstruct()` trial loop with parallel initialization
   - Create `_run_joint_optimization()` method
   - Modify `_compute_objective()` to work per-candidate

2. **[`base_attack.py`](file:///home/siladittyamanna/Documents/smanna/iisc/work1/breaching/breaching/attacks/base_attack.py)**
   - Update `_init_optimizer()` to accept list of tensors
   - Add optional `seed` parameter to `_initialize_data()`

3. **[`regularizers.py`](file:///home/siladittyamanna/Documents/smanna/iisc/work1/breaching/breaching/attacks/auxiliaries/regularizers.py)** (mostly done!)
   - `GroupRegularization` is already implemented
   - May need to ensure `aligner` is properly initialized

## Configuration Changes Needed

```yaml
# Example config addition
attack:
  restarts:
    num_trials: 8
    parallel_candidates: true  # NEW: enable parallel mode
  regularization:
    group_regularization:
      scale: 0.1
      warmup_iters: 500
      # aligner: ransac_flow  # or other alignment strategy
```