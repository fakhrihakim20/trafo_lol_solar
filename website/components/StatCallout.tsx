export default function StatCallouts({
  items,
}: {
  items: { value: string; label: string }[];
}) {
  return (
    <dl className="grid grid-cols-2 md:grid-cols-4 gap-y-8 gap-x-6 border-t border-b hairline py-8 my-12">
      {items.map((s) => (
        <div key={s.label}>
          <dt className="numera text-2xl md:text-3xl text-ink-950 dark:text-ink-50 tracking-tight">
            {s.value}
          </dt>
          <dd className="mt-2 text-xs leading-snug text-ink-600 dark:text-ink-300 max-w-[18ch]">
            {s.label}
          </dd>
        </div>
      ))}
    </dl>
  );
}
