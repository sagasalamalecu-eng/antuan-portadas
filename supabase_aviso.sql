-- Antuan Music · cuando la app encarga una portada nueva, avisa a GitHub para que la genere ya.
-- Sustituye PEGA_AQUI_TU_TOKEN por tu token de GitHub (solo con permiso sobre antuan-portadas) y ejecuta.
create extension if not exists pg_net with schema extensions;

create or replace function public.avisar_github_portada()
returns trigger
language plpgsql
security definer
set search_path = public, extensions
as $$
begin
  perform net.http_post(
    url := 'https://api.github.com/repos/sagasalamalecu-eng/antuan-portadas/dispatches',
    body := '{"event_type":"nueva-portada"}'::jsonb,
    headers := jsonb_build_object(
      'Authorization', 'Bearer PEGA_AQUI_TU_TOKEN',
      'Accept', 'application/vnd.github+json',
      'User-Agent', 'antuan-music',
      'Content-Type', 'application/json'
    )
  );
  return new;
end;
$$;

revoke all on function public.avisar_github_portada() from anon, authenticated;

drop trigger if exists avisar_github_portada on public.cover_loops;
create trigger avisar_github_portada
  after insert on public.cover_loops
  for each row when (new.status = 'queued')
  execute function public.avisar_github_portada();
