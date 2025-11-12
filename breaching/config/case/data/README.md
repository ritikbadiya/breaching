### Commom parameters

| Variable | Type           | Default Value | Description                                                                                               |
|----------|----------------|---------------|-----------------------------------------------------------------------------------------------------------|
| Name     | string         |               | Defaults generality to the config file name, and the name of the dataset itself                           |
|          |                |               |                                                                                                           |
| modality | enum           |               | "vision" or "text", based in the dataset nature                                                           |
|          |                |               |                                                                                                           |
| task     | enum           |               | Vision datasets: defaults to "classification".                                                            |
|          |                |               | Text datasets: can be "masked-lm" or  "causal-lm"                                                         |
|          |                |               |                                                                                                           |
|          |                |               |                                                                                                           |
| path     | string         | ~/data        | The path of the dataset in the system. Defaults to "~/data".                                              |
|          |                |               |                                                                                                           |
| size     | int            |               | The size of the data in the dataset                                                                       |
|          |                |               |                                                                                                           |
| shape    | 3-tuple or int |               | Vision datasets: it is a 3-tuple of (Channel, width, height) of the images on the dataset                 |
|          |                |               | Text datasets: it is the sequence_lenght of the dataset, that represents the number of tokens/words in it |

#### Federated Learning specifics:

| Variable            | Type | Default Value | Description                                   |
|---------------------|------|---------------|-----------------------------------------------|
| default_clients     | int  |               | Estimate on number of articles in dataset     |
|                     |      |               |                                               |
| partition           | enum |               | Can be "given", "balanced" or "unique-class". |
|                     |      |               | given: use natural data partition             |
|                     |      |               | balanced:                                     |
|                     |      |               | unique-class:                                 |
|                     |      |               |                                               |
| examples_from_split | enum |               | Can be "train", "training" or "validation".   |
|                     |      |               | train:                                        |
|                     |      |               | training:                                     |
|                     |      |               | validation:                                   |



#### Data-specific implementation constants:

| Variable   | Type    | Default Value | Description                                                                                           |
|------------|---------|---------------|-------------------------------------------------------------------------------------------------------|
| batch_size | int     |               |                                                                                                       |
|            |         |               |                                                                                                       |
| caching    | boolean | False         |                                                                                                       |
|            |         |               |                                                                                                       |
| db         | enum    | none          | Can be "none" or "LMDB"                                                                               |
|            |         |               | Database Setup. Use the cmd-line hydra overload feature to activate the LMDB module with data.db=LMDB |


### Vision datasets specific parameters:

| Variable | Type | Default Value | Description                                 |
|----------|------|---------------|---------------------------------------------|
| classes  | int  |               | The number of labels present in the dataset |

#### Preprocessing:

| Variable  | Type    | Default Value | Description |
|-----------|---------|---------------|-------------|
| normalize | boolean | True          |             |
|           |         |               |             |
| mean      | 3-tuple |               |             |
|           |         |               |             |
| std       | 3-tuple |               |             |

#### Data Augmentations:

| Variable                                 | Type   | Default Value | Description |
|------------------------------------------|--------|---------------|-------------|
| augmentations_train                      | object |               |             |
| augmentations_train.RandomCrop           | tuple  |               |             |
| augmentations_train.RandomHorizontalFlip | float  |               |             |
|                                          |        |               |             |
| augmentations_val                        | object |               |             |
| augmentations_val.Resize                 | int    |               |             |
| augmentations_val.CenterCrop             | int    |               |             |

### Text datasets specific parameters:

#### Only used when task=masked-lm:

| Variable       | Type    | Default Value | Description |
|----------------|---------|---------------|-------------|
| mlm_pobability | float   |               |             |
|                |         |               |             |
| disable_mlm    | boolean | False         |             |

#### Preprocessing:

| Variable   | Type | Default Value | Description |
|------------|------|---------------|-------------|
| tokenizer  | enum |               |             |
|            |      |               |             |
| vocab_size | int  |               |             |
