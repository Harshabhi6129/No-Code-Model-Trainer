-- Add display name to profiles so users can personalise their identity.
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS full_name text;
