# ZORL Autoresearch Runbook - 2026-06-03

## Start Long-Lived SGLang

Render one or more node-pinned SGLang controllers:

```bash
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-047.yaml
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-117.yaml
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-001.yaml
```

Launch after inspecting the rendered manifest:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --candidate experiments/zorl/autoresearch/candidates/SGL-047.yaml
```

The services are:

- `http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000`
- `http://zorl-ar-sglang-117.apanda.svc.cluster.local:30000`
- `http://zorl-ar-sglang-001.apanda.svc.cluster.local:30000`

Check readiness:

```bash
kubectl get pods -l autoresearch-id=sgl-047
kubectl logs -l autoresearch-id=sgl-047 --tail=80
kubectl run -it --rm zorl-sgl-probe --image=curlimages/curl --restart=Never -- \
  http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000/health_generate
```

## Launch A Candidate

Dry run:

```bash
python experiments/zorl/autoresearch/controller.py launch --id ZORL-001 --dry-run
python experiments/zorl/autoresearch/controller.py launch --id ZORL-OPD-000 --dry-run
```

Actual launch against the default `047` SGLang service:

```bash
python experiments/zorl/autoresearch/controller.py launch --id ZORL-001
```

Use another SGLang node:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-001 \
  --set-env INFER_URL=http://zorl-ar-sglang-117.apanda.svc.cluster.local:30000
```

## Monitor And Score

Tail the latest candidate result:

```bash
python experiments/zorl/autoresearch/controller.py monitor --idea-id ZORL-001 --result latest
```

Score without advancing the queue:

```bash
python experiments/zorl/autoresearch/controller.py score --idea-id ZORL-001 --result latest
```

Advance the queue after reviewing the scorecard:

```bash
python experiments/zorl/autoresearch/controller.py advance --id ZORL-001 --result latest
```

Scorecards are written under `experiments/zorl/autoresearch/scorecards/`.

## ZORL-OPD / ZORL-OPSD

The first OPD path is wired through the standalone SGLang client:

- `ZORL-OPD-000`: fast 2-digit OPD-format multiplication smoke.
- `ZORL-OPD-001`: 3-digit OPD multiplication science run gated by the smoke.
- Both use `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml`
  against the same SGL-001/047/117 controller pool.

Launch and score the smoke first:

```bash
python experiments/zorl/autoresearch/controller.py launch --id ZORL-OPD-000
python experiments/zorl/autoresearch/controller.py score --idea-id ZORL-OPD-000 --result latest
python experiments/zorl/autoresearch/controller.py advance --id ZORL-OPD-000 --result latest
```

For ZORL-OPSD, use the same pattern once its runnable task or trainer
driver exists: candidate YAML, `INFER_URL`, result root, and scoring gates.
