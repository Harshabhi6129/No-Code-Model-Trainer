-- Phase A3: Pipeline checkpoint/resume
-- Allows the /chat endpoint to persist the AgentContext after each stage
-- so that expensive training runs can resume without re-running earlier stages.

ALTER TABLE public.runs
  -- Full serialised AgentContext snapshot (jsonb — efficient partial updates)
  ADD COLUMN IF NOT EXISTS pipeline_checkpoint jsonb DEFAULT '{}',
  -- Which pipeline stages completed successfully (for fast skip-check on resume)
  ADD COLUMN IF NOT EXISTS completed_stages    text[]  DEFAULT '{}';

-- Index for fast lookup when loading a checkpoint by run_id
CREATE INDEX IF NOT EXISTS runs_completed_stages_idx
  ON public.runs USING GIN (completed_stages);

COMMENT ON COLUMN public.runs.pipeline_checkpoint IS
  'Serialised AgentContext stored after each stage completes. Used by resume logic.';

COMMENT ON COLUMN public.runs.completed_stages IS
  'Array of agent names (e.g. ["Intent","Data","Clean","Model"]) that finished successfully.';
