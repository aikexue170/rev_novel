# B200 on-demand rental prices vs our cost per 1,000 answers

Researched 2026-09-23 from providers' own public pricing pages where they render; third-party price trackers only where noted. Prices are USD per GPU-hour for one NVIDIA B200 (or the per-GPU share of a multi-GPU node where no single-GPU size exists). No accounts were created and no data entered.

## 1. Provider prices (cheapest first)

| # | Provider | $/GPU-h | Billing model | Source | Read on | Notes |
|---|---|---|---|---|---|---|
| 1 | RunPod Community Cloud | 5.98 | VM/pod, per-second billing | https://www.runpod.io/pricing | 2026-09-23 | 1x B200 pod; community hosts. Third-party trackers showed it out of stock on 2026-09-23. |
| 2 | Hyperbolic | 5.99 | VM (marketplace), hourly | https://www.hyperbolic.ai/marketplace | 2026-09-23 | Listed as "from $5.99/hr"; pricing refreshed weekly from suppliers. |
| 3 | Hyperstack | 6.00 | VM, per-minute billing (prepaid) | https://www.hyperstack.cloud/gpu-pricing | 2026-09-23 | Not on the requested list; included as a cross-check. |
| 4 | Modal (GPU only) | 6.25 | Serverless, per-second | https://modal.com/pricing | 2026-09-23 | $0.001736/s. Excludes host CPU/RAM. |
| 5 | Verda (formerly DataCrunch) | 6.62 | VM, on-demand | https://verda.com/pricing | 2026-09-23 | 1x B200 SXM6 180GB. Spot $3.31/h. datacrunch.io redirects to verda.com. |
| 6 | Lambda (8x node, per GPU) | 6.69 | VM, on-demand | https://lambda.ai/service/gpu-cloud | 2026-09-23 | $53.52/h for 8x; per-GPU share shown. Trackers showed out of stock 2026-09-23. |
| 7 | RunPod Secure Cloud | 6.79 | VM/pod, per-second billing | https://www.runpod.io/pricing | 2026-09-23 | 1x B200 pod in RunPod-operated datacenters. |
| 8 | Vast.ai (market median) | 6.82 | VM (marketplace), on-demand | https://vast.ai/pricing/gpu/B200 | 2026-09-23 | Live page showed no 1x B200 offers when read; page title advertises "from $7.38/hr". $6.82 is the Sep-18 median from thundercompute.com; computeprices.com showed $6.25-8.13 on Sep 22-23. Interruptible listings far lower. |
| 9 | Lambda (1x instance) | 6.99 | VM, on-demand | https://lambda.ai/service/gpu-cloud | 2026-09-23 | Single-GPU B200 SXM6 instance. |
| 10 | Nebius | 7.15 | VM, on-demand | https://nebius.com/prices | 2026-09-23 | HGX B200, 20 vCPU + 224 GB RAM per GPU; multi-GPU configs only, no 1x listed. Spot ~$3.95 per trackers. |
| 11 | Modal (all-in: GPU + 8 cores + 96 GiB) | 7.39 | Serverless, per-second | https://modal.com/pricing | 2026-09-23 | Our actual config: $6.25 GPU + $0.377 CPU (8 x $0.0000131/s) + $0.767 RAM (96 x $0.00000222/s). |
| 12 | Google Cloud A4 Flex-start (8x node, per GPU) | 8.06 | VM, DWS Flex-start (not standard on-demand) | https://cloud.google.com/products/compute/pricing/accelerator-optimized | 2026-09-23 | a4-highgpu-8g: on-demand price is N/A; Flex-start $64.44/h, Calendar Mode $90.22/h, Spot $39.63/h per node. No 1x size. |
| 13 | CoreWeave HGX B200 (8x node, per GPU) | 8.60 | VM (Kubernetes), on-demand | https://www.coreweave.com/pricing | 2026-09-23 | $68.80/h per 8-GPU instance. No 1x size. |
| 14 | Together AI (dedicated) | 8.99 | Dedicated endpoint, hourly | https://www.together.ai/pricing | 2026-09-23 | Dedicated inference, single-tenant. |
| 15 | Baseten (dedicated) | 9.98 | Serverless-style dedicated deployment, per-minute | https://www.baseten.co/pricing/ | 2026-09-23 | $0.16633/min. 180 GiB VRAM B200. |
| 16 | AWS P6-B200 (8x node, per GPU) | 14.24 | VM, on-demand | https://instances.vantage.sh/aws/ec2/p6-b200.48xlarge | 2026-09-23 | p6-b200.48xlarge $113.9328/h us-east-1 Linux (AWS pricing via vantage.sh/doit.com; AWS instance page lists no price). Spot $42.35/h. No 1x size. |
| 17 | Azure ND GB200 v6 (4x node, per GPU) | 27.04 | VM, pay-as-you-go | https://prices.azure.com/api/retail/prices?$filter=contains(armSkuName,'B200') | 2026-09-23 | Standard_ND128isr_NDR_GB200_v6 $108.16/h eastus Linux; 4 Blackwell GPUs (GB200, not plain B200). Azure lists no B200-only VM. |

Not priced:
- Crusoe: B200 listed as "Contact sales" for on-demand and spot; no public price. (https://www.crusoe.ai/cloud/pricing)
- Lightning AI: Pricing page is client-rendered and returned only a loading stub on 2026-09-23; B200 price could not be verified (third-party trackers list H200 at $6.53/h max single-GPU, no B200). (https://lightning.ai/pricing/)
- Google Cloud (standard on-demand): a4-highgpu-8g on-demand price is N/A; only Flex-start, Calendar Mode, Spot and CUDs. (https://cloud.google.com/products/compute/pricing/accelerator-optimized)

## 2. Cost per 1,000 answers at each price

Throughput at full load from `summary.json` (best `answers_per_second` row per arm, all measured on Modal's B200 host): Qwen3.5-4B 93.25 ans/s (concurrency 128), Qwen3.5-9B 69.24 ans/s (concurrency 256), Qwen3.8-27B 25.08 ans/s (concurrency 128). Formula: price/h / 3600 / answers_per_s x 1000. Jev's billed price is $0.0366 per 1,000 answers; "vs Jev" = 0.0366 / ours (>1 means we are cheaper).

| Provider | $/GPU-h | Qwen3.5-4B $/1k | vs Jev | Qwen3.5-9B $/1k | vs Jev | Qwen3.8-27B $/1k | vs Jev |
|---|---|---|---|---|---|---|---|
| RunPod Community Cloud | 5.98 | 0.0178 | 2.05x | 0.0240 | 1.53x | 0.0662 | 0.55x |
| Hyperbolic | 5.99 | 0.0178 | 2.05x | 0.0240 | 1.52x | 0.0663 | 0.55x |
| Hyperstack | 6.00 | 0.0179 | 2.05x | 0.0241 | 1.52x | 0.0665 | 0.55x |
| Modal (GPU only) | 6.25 | 0.0186 | 1.97x | 0.0251 | 1.46x | 0.0692 | 0.53x |
| Verda (formerly DataCrunch) | 6.62 | 0.0197 | 1.86x | 0.0266 | 1.38x | 0.0733 | 0.50x |
| Lambda (8x node, per GPU) | 6.69 | 0.0199 | 1.84x | 0.0268 | 1.36x | 0.0741 | 0.49x |
| RunPod Secure Cloud | 6.79 | 0.0202 | 1.81x | 0.0272 | 1.34x | 0.0752 | 0.49x |
| Vast.ai (market median) | 6.82 | 0.0203 | 1.80x | 0.0274 | 1.34x | 0.0755 | 0.48x |
| Lambda (1x instance) | 6.99 | 0.0208 | 1.76x | 0.0280 | 1.31x | 0.0774 | 0.47x |
| Nebius | 7.15 | 0.0213 | 1.72x | 0.0287 | 1.28x | 0.0792 | 0.46x |
| Modal (all-in: GPU + 8 cores + 96 GiB) | 7.39 | 0.0220 | 1.66x | 0.0296 | 1.23x | 0.0818 | 0.45x |
| Google Cloud A4 Flex-start (8x node, per GPU) | 8.06 | 0.0240 | 1.52x | 0.0323 | 1.13x | 0.0893 | 0.41x |
| CoreWeave HGX B200 (8x node, per GPU) | 8.60 | 0.0256 | 1.43x | 0.0345 | 1.06x | 0.0953 | 0.38x |
| Together AI (dedicated) | 8.99 | 0.0268 | 1.37x | 0.0361 | 1.01x | 0.0996 | 0.37x |
| Baseten (dedicated) | 9.98 | 0.0297 | 1.23x | 0.0400 | 0.91x | 0.1105 | 0.33x |
| AWS P6-B200 (8x node, per GPU) | 14.24 | 0.0424 | 0.86x | 0.0571 | 0.64x | 0.1577 | 0.23x |
| Azure ND GB200 v6 (4x node, per GPU) | 27.04 | 0.0805 | 0.45x | 0.1085 | 0.34x | 0.2995 | 0.12x |

## 3. Caveats

All figures are list on-demand prices (spot/interruptible tiers are 40-80% cheaper - Verda spot $3.31, Nebius spot ~$3.95, Google Spot $4.95/GPU, AWS Spot $5.29/GPU - but preemptible and unsuitable for a latency SLA). Availability is the real constraint: on 2026-09-23 third-party trackers showed RunPod Community, Lambda and Nebius B200 1x capacity out of stock, and Vast.ai had no live 1x B200 offer, so the cheapest list price is often not rentable on demand. Hyperscalers (AWS, Google, Azure) and CoreWeave sell only 4- or 8-GPU nodes, so the per-GPU figure assumes we could fill a whole node; Google's A4 has no standard on-demand tier at all. Prices exclude egress (typically $0.05-0.09/GB on hyperscalers, free or near-free on Modal/RunPod/Lambda/Verda), storage, and any minimum commitment (Together/Baseten/CoreWeave quote reserved discounts on contact; Hyperstack is prepaid). Other providers' prices include the host CPU and RAM, so the like-for-like comparison against Modal is the $7.39/h all-in figure, not $6.25. Finally, our throughput numbers were measured on Modal's B200 host; we saw a 2x single-request latency difference between two hosts, so host CPU, memory bandwidth and driver stack matter and answers/s on another provider's host could differ materially in either direction - it must be re-measured there before any cost claim is made.

## 4. Summary

Cheapest reputable on-demand B200 found: RunPod Community Cloud at $5.98/GPU-h (Hyperbolic $5.99 and Hyperstack $6.00 are within 2 cents; Modal GPU-only is $6.25, so Modal is already within 5% of the market floor and cheaper than Lambda, Nebius, CoreWeave and every hyperscaler).
At $5.98/h our cost is $0.0178 per 1k answers for Qwen3.5-4B (2.1x cheaper than Jev's $0.0366) and $0.0662 per 1k for Qwen3.8-27B (0.55x, i.e. ~81% more expensive than Jev); on Modal all-in ($7.39/h) the same numbers are $0.0220 and $0.0818.
