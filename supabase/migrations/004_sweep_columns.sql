-- Add hyperparameter sweep columns to runs table.
-- sweep_id groups all child runs launched in one sweep.
-- parent_run_id links back to the completed run the sweep was derived from.
-- sweep_config stores the specific param combo this child run used.

ALTER TABLE runs
  ADD COLUMN IF NOT EXISTS sweep_id       uuid,
  ADD COLUMN IF NOT EXISTS parent_run_id  uuid REFERENCES runs(id),
  ADD COLUMN IF NOT EXISTS sweep_config   jsonb;

CREATE INDEX IF NOT EXISTS runs_sweep_id_idx ON runs (sweep_id) WHERE sweep_id IS NOT NULL;
