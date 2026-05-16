// Editorial pull-quote: italic Newsreader serif, left hairline rule, narrow measure.
// Placed between body prose and chart in each finding.

export default function PullQuote({ children }: { children: React.ReactNode }) {
  return (
    <blockquote className="pull-quote">
      {children}
    </blockquote>
  );
}
