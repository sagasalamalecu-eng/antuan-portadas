-- Antuan Music · cuando un móvil pide una canción, avisa a GitHub para que la baje ya (sin el PC).
-- Pegar entero en Supabase → SQL Editor → Run. Reutiliza el token de GitHub que ya está en el aviso de las
-- portadas (supabase_aviso.sql): lo guarda en la bóveda de Supabase, así no hay que volver a pegarlo.
create extension if not exists pg_net with schema extensions;

do $$
declare
  src text;
  tok text;
begin
  if not exists (select 1 from vault.secrets where name = 'github_token') then
    select pg_get_functiondef('public.avisar_github_portada()'::regprocedure) into src;
    tok := substring(src from 'Bearer ([A-Za-z0-9_]+)');
    if tok is null or tok = 'PEGA_AQUI_TU_TOKEN' then
      raise exception 'No encuentro el token de GitHub en avisar_github_portada()';
    end if;
    perform vault.create_secret(tok, 'github_token', 'GitHub: avisos de Antuan Music');
  end if;
end $$;

create or replace function public.avisar_github_cancion()
returns trigger
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  tok text;
begin
  select decrypted_secret into tok from vault.decrypted_secrets where name = 'github_token';
  perform net.http_post(
    url := 'https://api.github.com/repos/sagasalamalecu-eng/antuan-portadas/dispatches',
    body := '{"event_type":"nueva-cancion"}'::jsonb,
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || tok,
      'Accept', 'application/vnd.github+json',
      'User-Agent', 'antuan-music',
      'Content-Type', 'application/json'
    )
  );
  return new;
end;
$$;

revoke all on function public.avisar_github_cancion() from public, anon, authenticated;

drop trigger if exists avisar_github_cancion on public.song_files;
create trigger avisar_github_cancion
  after insert or update of status on public.song_files
  for each row when (new.status = 'queued')
  execute function public.avisar_github_cancion();
