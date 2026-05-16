// Masthead row at the top of the piece. Hairline-bordered top + bottom,
// uppercase tracked-out mono, no decoration — magazine convention.

const ITEMS = [
  'Fakhri Hakim',
  'PT PLN (Persero)',
  'Submitted March 2026',
  '12-min read',
];

export default function Byline() {
  return (
    <div className="byline">
      <div className="max-w-page mx-auto px-6 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        {ITEMS.map((item, i) => (
          <span key={item} className={i === 0 ? 'text-ink-900 dark:text-ink-100' : ''}>
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}
