function normalizeHttpUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) return 'http://127.0.0.1:8000';
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  const scheme = /^(localhost|127\.0\.0\.1)(:\d+)?$/i.test(trimmed) ? 'http' : 'https';
  return `${scheme}://${trimmed}`;
}

const rawApiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) || 'http://127.0.0.1:8000';
export const API_BASE_URL = normalizeHttpUrl(rawApiBase);

const rawWsUrl = (import.meta.env.VITE_WS_URL as string | undefined)?.trim();
export const WS_CLASSROOM_URL = rawWsUrl
  ? rawWsUrl.replace(/\/+$/, '')
  : `${API_BASE_URL.replace(/^http/i, 'ws')}/ws/classroom`;

