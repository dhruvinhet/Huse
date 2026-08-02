-- Run this once in Supabase Dashboard > SQL Editor.
create table if not exists public.generation_runs (
  id uuid primary key default gen_random_uuid(),
  topic text not null,
  status text not null default 'queued'
    check (status in ('queued', 'running', 'completed', 'failed')),
  output_path text,
  error_message text,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

alter table public.generation_runs enable row level security;
