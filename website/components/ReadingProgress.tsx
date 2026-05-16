'use client';

import { useEffect, useRef } from 'react';

// Thin fixed-top bar showing how far down the piece the reader is.
// Drives a CSS variable via requestAnimationFrame (per taste-skill §5
// "Hardware Acceleration" — never `addEventListener('scroll')` for
// continuous reads). Respects prefers-reduced-motion.

export default function ReadingProgress() {
  const fillRef = useRef<HTMLSpanElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return; // no continuous animation in reduced-motion mode

    const tick = () => {
      const el = fillRef.current;
      if (!el) return;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const pct = max > 0 ? Math.min(100, Math.max(0, (window.scrollY / max) * 100)) : 0;
      el.style.setProperty('--progress', `${pct}%`);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <div className="reading-progress" aria-hidden>
      <span ref={fillRef} />
    </div>
  );
}
