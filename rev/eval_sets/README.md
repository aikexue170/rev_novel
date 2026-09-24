# Frozen evaluation sets (never used for training)

- `jev_official_262.jsonl`: 262 rows, sha256 `e1a4e070a9bc11f5a81d0827168ed92f22cf56a6045b07498480f38eda6820d4`, sources {'typesafe_invoice_processing': 29, 'typesafe_agent_trace_observability': 28, 'typesafe_customer_service': 24, 'typesafe_security_incidents': 21, 'composition_holdout': 20, 'emotion': 20, 'legacy_holdout': 20, 'mmlu': 20, 'paws': 20, 'qnli': 20, 'sciq': 20, 'tweet_offensive': 20}
- `public_holdout_975.jsonl`: 975 rows, sha256 `e743dfc0b4ab614c65535c759c6d234ccd45b8ab26c8d99e1bea5bd2ab6ba93c`, sources {'contractnli': 175, 'race': 200, 'cosmosqa': 200, 'multirc': 200, 'socialiqa': 200}

`jev_official_262` = Jev's 102 published workflow questions (evals.typesafe.ai: invoices, agent traces, customer service, security) + a 160-question public-task slice; labels are the published reference answers. `public_holdout_975` = the ContractNLI/RACE/CosmosQA/MultiRC/Social IQa holdout. Training scripts assert no id/group/state overlap with these files.

## Sources and licenses

`public_holdout_975.jsonl` is built from five public datasets and redistributed here so the benchmark can be rerun exactly: ContractNLI (CC BY 4.0), CosmosQA (CC BY 4.0), Social IQa (CC BY 4.0), MultiRC from SuperGLUE (research use, see the SuperGLUE terms), and RACE, which its authors license for non-commercial research use only. If you use this file, those terms apply to those rows. Labels are the datasets' own; nothing was relabelled.

`jev_official_262.jsonl` contains the 102 workflow questions TypeSafe published for Jev (their questions and their reference labels, copied from their public eval page) plus a 160-question slice of public task families. It is kept so the numbers in `results/` can be reproduced with one command; the questions belong to TypeSafe.
