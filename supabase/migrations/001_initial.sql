-- ModelForge initial schema

create table public.profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  email text unique not null,
  full_name text,
  avatar_url text,
  created_at timestamptz default now() not null
);

create table public.runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  status text not null default 'pending'
    check (status in ('pending', 'running', 'completed', 'failed', 'cancelled')),
  task_type text,
  model_id text,
  dataset_filename text,
  dataset_rows integer,
  intent_spec jsonb default '{}',
  model_recipe jsonb default '{}',
  metrics jsonb default '{}',
  artifact_path text,
  hf_model_url text,
  error_message text,
  created_at timestamptz default now() not null,
  completed_at timestamptz
);

create table public.run_events (
  id bigserial primary key,
  run_id uuid references public.runs(id) on delete cascade not null,
  event_type text not null check (event_type in ('agent', 'progress', 'metric', 'log', 'error', 'done')),
  data jsonb not null default '{}',
  created_at timestamptz default now() not null
);

-- Indexes
create index runs_user_id_idx on public.runs(user_id);
create index runs_status_idx on public.runs(status);
create index run_events_run_id_idx on public.run_events(run_id);

-- RLS
alter table public.profiles enable row level security;
alter table public.runs enable row level security;
alter table public.run_events enable row level security;

create policy "Users can view own profile"   on public.profiles for select using (auth.uid() = id);
create policy "Users can update own profile" on public.profiles for update using (auth.uid() = id);
create policy "Users can insert own profile" on public.profiles for insert with check (auth.uid() = id);

create policy "Users can view own runs"   on public.runs for select using (auth.uid() = user_id);
create policy "Users can insert own runs" on public.runs for insert with check (auth.uid() = user_id);
create policy "Users can update own runs" on public.runs for update using (auth.uid() = user_id);

create policy "Users can view own run events" on public.run_events
  for select using (exists (select 1 from public.runs where id = run_id and user_id = auth.uid()));
create policy "Users can insert own run events" on public.run_events
  for insert with check (exists (select 1 from public.runs where id = run_id and user_id = auth.uid()));

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email);
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
