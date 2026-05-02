-- Add deploy tracking columns to runs
alter table public.runs
  add column if not exists model_card    text,
  add column if not exists deploy_status text default 'not_deployed'
    check (deploy_status in ('not_deployed', 'deploying', 'deployed', 'failed', 'skipped')),
  add column if not exists hf_repo_id   text;
