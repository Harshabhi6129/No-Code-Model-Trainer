-- Cross-device sync for the train workspace (issue #28).
-- One row per user holding their session list as a JSON blob. localStorage stays
-- the primary store on each device; this row is the cloud backup used to restore
-- sessions on a new device or after a cache clear.

create table if not exists public.training_sessions (
  user_id    uuid references auth.users(id) on delete cascade primary key,
  data       jsonb       not null default '[]',
  updated_at timestamptz not null default now()
);

alter table public.training_sessions enable row level security;

create policy "Users can view own training sessions"
  on public.training_sessions for select using (auth.uid() = user_id);
create policy "Users can insert own training sessions"
  on public.training_sessions for insert with check (auth.uid() = user_id);
create policy "Users can update own training sessions"
  on public.training_sessions for update using (auth.uid() = user_id);
