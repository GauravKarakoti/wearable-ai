# Test Phase: preparing your submission image

In the Test Phase you submit a **container image** instead of a predictions file.
Your model runs on organizer hardware against the held-out test split, so
everything it needs must be inside the image.

Three steps: build, validate, submit.

---

## 1. Prepare your image

Start from the `Containerfile` in this directory. It already contains the
evaluation stack (CUDA 12.8, Python 3.10, PyTorch 2.10.0, vLLM 0.19.1) and the
starter kit. Edit only the marked extension block at the bottom.

### Add your weights

```dockerfile
COPY my_weights/ /models/my_model/
```

Weights **must be baked into the image**. Evaluation nodes cannot reach your
storage, and downloading at run time would count against your per-turn budget.

### Register your model

Add your model class to `MODEL_REGISTRY` in `model.py` (see *Using Your Own
Model* in the starter kit README), then reference it with
`--model-type <your_key>`.

### Four rules that will fail your submission if broken

1. **Keep `bash` in the image.** The job scheduler runs a bash script inside your
   container before your code starts. A `distroless`, `busybox`, or `-slim` base
   without bash fails immediately with exit code 127 and produces no output.
2. **Do not hardcode dataset paths.** Organizers pass `--video-folder`,
   `--golden`, and `--predictions` at evaluation time. The relative paths that
   work against a local clone (`../egolongqa/val/`) will not exist.
3. **Do not rely on `ENTRYPOINT`.** The runtime invokes `run_evaluation.py`
   directly; any `ENTRYPOINT` you set is ignored.
4. **Fit on one node.** One copy of your model must run within a single node of
   8 x H100 80 GB. We run up to 16 nodes to get through the split faster, but
   that is parallelism across queries: each node runs its own full copy of your
   model, so one node's 640 GB is the memory a copy may use. We do not shard a
   model across nodes. See *Resource limits* below and contact us before you
   build if yours needs more.

### Build

```bash
./build_image.sh                        # tags wearable-ai-eval:latest
./build_image.sh --tag my-team:v1       # custom tag
```

---

## 2. Validate before you submit

```bash
./validate_image.sh                     # or: ./validate_image.sh my-team:v1
```

This checks every requirement above: bash present, CUDA-enabled torch, starter
kit intact, `/models` populated, model class registered. **Organizers run this
exact script when your image arrives**, so a pass here is a pass there.

Expected output:

```
Validating wearable-ai-eval:latest

Runtime prerequisites
  PASS  /bin/bash exists (required by the evaluation runtime)
  PASS  python3 present
...
All 12 checks passed -- image satisfies the submission contract.
```

On the template as shipped, before you add anything, 11 of the 12 pass and the
failure is `/models is non-empty`. That one is what your weights fix.

### Test it end-to-end yourself

Validation checks structure, not correctness. To confirm your model actually
produces predictions, run it on the public **val** split:

Run it from this directory. `../..` is the repository root, which is where the
val videos and gold files live, and `out/` has to exist before it is mounted:

```bash
mkdir -p out
podman run --rm --device nvidia.com/gpu=all \
  -v "$PWD/../..:/data:ro" -v "$PWD/out:/output" \
  wearable-ai-eval:latest \
  python3 /app/run_evaluation.py \
    --task longqa --model-type my_model \
    --llm-model /models/my_model \
    --video-folder /data/egolongqa/val \
    --golden /data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
    --predictions /output/predictions.jsonl \
    --max-samples 5
```

Start with `--max-samples 5`. If that produces a well-formed
`predictions.jsonl`, the full run will too.

---

## 3. Submit

Registering your team on the leaderboard gives you an access key and the URI of a
private repository created for you. Log in to it, then push:

```bash
export AWS_ACCESS_KEY_ID=...        # both shown when you register
export AWS_SECRET_ACCESS_KEY=...

aws ecr get-login-password --region us-east-2 \
  | podman login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com

./build_image.sh --tag my-team:v1 \
  --push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/<your-team>
```

Pass the repository URI **complete**. Only the version (`v1`) carries over from
`--tag`; the repository is the one issued to you and cannot be changed.

The script prints the digest it pushed. **Register that digest**, not the tag:
that is the exact image we score.

Use a new version for each build (`:v1`, `:v2`, ...). Tags are immutable, so an
existing one cannot be overwritten.

---

## Resource limits

| Limit | Value |
| --- | --- |
| Time per generation / turn | 300 s |
| GPUs one copy of your model may use | 8 x H100 80 GB on a single node, 640 GB |
| Evaluation parallelism | up to 16 nodes, run by the organizers |

Exceeding the per-turn timeout aborts that turn and scores it as empty.

**We provide up to 16 nodes, and your model must still fit on one.** The 16 nodes
are there to get through the test split faster: the queries are divided between
them and **each node runs its own full, independent copy of your image**. So the
budget for a single copy is one node, 8 x H100 80 GB. We do not shard one model
across nodes. If yours genuinely needs more than one node to hold a single copy,
write to the organizers **before you build**: that requires a custom image and
our agreement, and it is not something we can arrange after a submission arrives.

Nothing about the node count is baked into your image, so you can build and test
on whatever you have, and one GPU is fine. Resolve the GPU count at run time rather
than hardcoding it: `run_evaluation.py --num-gpus` is passed to your model, and
with no value it uses every GPU it can see.

## Reproducibility

The `Containerfile` pins its base image by digest rather than tag. Please keep
it that way: tags are mutable upstream, and an unpinned base means your build
and the organizers' validation can silently differ.
