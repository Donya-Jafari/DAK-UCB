# DAK-UCB

Diversity-Aware Prompt Routing for LLMs and Generative Models.

This repo contains a modularized, minimal implementation of the DAK-UCB / Mixture-DAK-UCB algorithm.

## Quick Start

```bash
python -m dak_ucb.main --config config.yaml
```

## Notes
- Model generation and embedding extraction are intentionally **placeholders**. Configure your own backends in `dak_ucb/models.py` and `dak_ucb/embeddings.py`.
- A `use_mock_data` flag in `config.yaml` lets you run the full loop with random embeddings.
