# llm_forensics.toolkit

Measurements for looking for ROME/MEMIT edits by hand. Each module takes any weights that
`llmf scan` accepts, writes JSON (and sometimes `.pt`) to `--out` (default
`toolkit-out/<model>/`) and prints a table. None of them decides whether a model is edited.
Each module's docstring gives its outputs, an estimated cost on a 7B model on a 24 GB GPU, and
when its numbers mean something (`python -c "import llm_forensics.toolkit.trace as m; help(m)"`).

| Module | Measures | Needs |
|---|---|---|
| `spectra` | per-layer top-k singular values, ratios, participation ratios, column-norm z, for every weight family | weights |
| `corpus` | a text corpus sampled from the model, cached per model under `<out>/corpus/` | generation |
| `selectivity` | max/sd and max/p99 of corpus keys projected on each layer's top-k MLP-out components | corpus |
| `whiten` | one layer's key covariance; the subject's whitened key direction against random ones | corpus, subject |
| `trace` | P(target) with each layer's MLP output zeroed or mean-ablated at one position | prompt |
| `surgery` | outputs, token diff over relation templates and NLL with one singular triplet removed | layer, component |
| `probe` | greedy completions and next-token probabilities over a prompt file | prompts |

Common flags: `--out DIR`, `--device`, `--dtype`, `--base` (for a delta whose metadata names no
base). Corpus users also take `--corpus-tokens`, `--seq-len`, `--gen-batch`, `--seed`,
`--batch-size`; the same settings reuse the cached corpus.

```bash
M=~/.cache/llm_forensics/models/<bucket>/<suspect>      # any weights spec

python -m llm_forensics.toolkit.spectra     $M --k 5 --families mlp.down_proj
python -m llm_forensics.toolkit.corpus      $M --corpus-tokens 30000
python -m llm_forensics.toolkit.selectivity $M --k 48 --top 20
python -m llm_forensics.toolkit.probe       $M --prompts facts.jsonl
python -m llm_forensics.toolkit.trace       $M --prompt "The Eiffel Tower is located in" \
    --subject "Eiffel Tower" --target " Rome"
python -m llm_forensics.toolkit.whiten      $M --layer 17 --subject "Eiffel Tower" --save-cov
python -m llm_forensics.toolkit.surgery     $M --layer 17 --index 2 --subject "Eiffel Tower" \
    --prompts facts.jsonl --text-file outside.txt
```

Prompt files hold one prompt per line, or JSON lines `{"prompt": ..., "target": ..., "subject": ...}`;
`#` lines are skipped.

From Python, every module has a function taking a `LoadedModel` (or a `(model, tokenizer)` pair):

```python
from llm_forensics.core.loading import resolve_weights
from llm_forensics.toolkit import corpus, selectivity, surgery

lm = resolve_weights(M)
c = corpus.load_or_generate(lm, "toolkit-out/suspect")
res = selectivity.measure(lm, c, k=48)
with surgery.component_edited(lm.model, layer=17, index=2):    # restored on exit
    print(lm.tok.decode(lm.model.generate(lm.tok("The Eiffel Tower is in", return_tensors="pt")
                                          .input_ids.to(lm.device), max_new_tokens=8)[0]))
```

Activations are streamed batch by batch and position 0 (the attention sink) is left out of
every corpus statistic. `whiten` works on one layer per call.
