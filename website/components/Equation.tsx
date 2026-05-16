'use client';

import 'katex/dist/katex.min.css';
import katex from 'katex';
import { useMemo } from 'react';

export default function Equation({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'html' }),
    [tex, display],
  );
  return (
    <span
      className={display ? 'block my-6 overflow-x-auto text-ink-900 dark:text-ink-100' : 'inline'}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
