'use client';

import { useEffect, useState } from 'react';

// Two-state toggle: Light ↔ Dark. On first visit before any click, the page
// follows the OS preference (set by the inline preload script in layout.tsx).
// Once the user clicks, the choice is saved to localStorage.theme and
// overrides the OS preference on subsequent visits / pages.

export default function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  // Read the initial state once, after mount, to avoid SSR/hydration mismatch.
  // The .dark class is set by the inline preload script before paint.
  useEffect(() => {
    setDark(document.documentElement.classList.contains('dark'));
  }, []);

  const toggle = () => {
    const next = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', next);
    try {
      localStorage.setItem('theme', next ? 'dark' : 'light');
    } catch {
      /* localStorage unavailable (privacy mode) — toggle still works for this session */
    }
    setDark(next);
  };

  // Render a placeholder square pre-hydration so layout doesn't shift.
  if (dark === null) {
    return (
      <span
        aria-hidden
        className="inline-block w-8 h-8 rounded-md border hairline opacity-60"
      />
    );
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      onClick={toggle}
      className="inline-flex items-center justify-center w-9 h-9 rounded-md border hairline text-ink-700 dark:text-ink-200 hover:bg-ink-100 dark:hover:bg-ink-900 transition-colors active:scale-[0.96]"
    >
      {/* Sun (shown in dark mode → click flips to light) / Moon (shown in light → click flips to dark) */}
      {dark ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
