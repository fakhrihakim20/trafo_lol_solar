import type { Metadata } from 'next';
import { GeistSans } from 'geist/font/sans';
import { JetBrains_Mono, Newsreader } from 'next/font/google';
import './globals.css';
import SectionNav from '@/components/SectionNav';
import ReadingProgress from '@/components/ReadingProgress';

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

// Runs synchronously before first paint to set .dark on <html> based on the
// reader's saved preference (or OS default). Without this, the page would
// briefly flash in the wrong theme on every load.
const themePreloadScript = `(function(){try{var s=localStorage.getItem('theme');var d=s?s==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;if(d)document.documentElement.classList.add('dark');}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${geist.variable} ${news.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themePreloadScript }} />
      </head>
      <body className="font-sans antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:bg-ink-950 focus:text-ink-50 focus:px-3 focus:py-2 focus:rounded"
        >
          Skip to content
        </a>
        <ReadingProgress />
        <SectionNav />
        <main id="main" className="relative z-[2]">
          {children}
        </main>
      </body>
    </html>
  );
}
