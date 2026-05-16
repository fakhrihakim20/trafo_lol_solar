import type { Metadata } from 'next';
import { GeistSans } from 'geist/font/sans';
import { JetBrains_Mono, Newsreader } from 'next/font/google';
import './globals.css';
import SectionNav from '@/components/SectionNav';

const geist = GeistSans;
const news = Newsreader({
  subsets: ['latin'],
  variable: '--font-newsreader',
  display: 'swap',
  weight: ['300', '400', '500'],
  style: ['normal', 'italic'],
});
const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Transformer Loss-of-Life under Rooftop PV · GI Jember · PT PLN',
  description:
    'A stakeholder companion to the IEEE ICT-PEP 2026 paper on transformer aging at GI Jember under rooftop-PV penetration, climate warming, and EV load growth.',
  authors: [{ name: 'Fakhri Hakim, PT PLN (Persero)' }],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${news.variable} ${mono.variable}`}>
      <body className="font-sans antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:bg-ink-950 focus:text-ink-50 focus:px-3 focus:py-2 focus:rounded"
        >
          Skip to content
        </a>
        <SectionNav />
        <main id="main" className="relative z-[2]">
          {children}
        </main>
        <footer className="relative z-[2] border-t hairline mt-32 py-12 px-6">
          <div className="max-w-page mx-auto flex flex-col md:flex-row md:items-end md:justify-between gap-6 text-sm text-ink-600 dark:text-ink-300">
            <div>
              <p className="font-serif italic text-ink-700 dark:text-ink-200 text-base">
                Transformer Loss-of-Life under Rooftop-PV Penetration.
              </p>
              <p className="mt-1">Companion report to the IEEE ICT-PEP 2026 manuscript.</p>
            </div>
            <div className="md:text-right">
              <p>Fakhri Hakim · PT PLN (Persero) · 2026</p>
              <p className="mt-1">
                <a
                  className="underline underline-offset-4 hover:text-ink-900 dark:hover:text-ink-50"
                  href="https://github.com"
                >
                  Source &amp; reproducibility
                </a>
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
