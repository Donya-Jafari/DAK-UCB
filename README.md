# Mixture-DAK-UCB

## Dataset Layout

Put your datasets here:

- `input_dataset/`
- `target_dataset/`

Use `config.yaml` to point to them:

```yaml
data:
  input: input_dataset/images
  output: target_dataset/prompts.txt
  generated: generated_output
```

Meaning:

- `input`: the source dataset given to the selected model
- `output`: the reference target dataset used for evaluation
- `generated`: where model outputs are saved


## Mock vs Real Models

Mock mode is controlled here:

```yaml
runtime:
  use_mock: true
```

- `true`: uses random embeddings and mock outputs
- `false`: expects you to implement your real models

## Where To Add Your Models

There are 2 places to edit in `Mixture-DAK-UCB.py`.

1. Embedding models

Edit:

- `class EmbeddingBackend`
- `encode_input(...)`
- `encode_output(...)`

Right now these raise `NotImplementedError` when `use_mock` is `false`.

2. Generation or captioning models

Edit:

- `class ModelBackend`
- `run(...)`

Right now it returns mock outputs in mock mode and raises `NotImplementedError` otherwise.

Add your own generation logic there .

The list of available models is defined in `config.yaml`:

You can add your own model names there and branch on `self.name` inside `ModelBackend.run(...)`.

## How To Run

Run:

```bash
python3 Mixture-DAK-UCB.py --config config.yaml
```

Results are written to the file set in:

```yaml
runtime:
  result_file: results.json
```
