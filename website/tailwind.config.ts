import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './content/**/*.{ts,tsx}',
  ],
  darkMode: 'media',
  theme: {
    extend: {
      colors: {
        // Off-black, never #000 (taste-skill rule)
        ink: {
          50: '#FBFBFA',
          100: '#F7F6F3',
          200: '#EAEAEA',
          300: '#D5D4D0',
          400: '#9B9A95',
          500: '#787774',
          600: '#56544F',
          700: '#37352F',
          800: '#2F3437',
          900: '#1A1A1A',
          950: '#111111',
        },
        // Single saturated accent — PLN field engineering blue, desaturated below 80%
        signal: {
          50: '#F0F4FB',
          100: '#DDE5F2',
          300: '#9AAFD4',
          500: '#1B3F8B',
          600: '#163374',
          700: '#11295E',
        },
        // Diverging spot accents (heatmap, status) — washed pastels
        warm: '#FBF3DB',
        warmInk: '#956400',
        cool: '#E1F3FE',
        coolInk: '#1F6C9F',
        leaf: '#EDF3EC',
        leafInk: '#346538',
      },
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['var(--font-newsreader)', 'ui-serif', 'Georgia', 'serif'],
        mono: ['var(--font-jetbrains)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      maxWidth: {
        prose: '68ch',
        chart: '1080px',
        page: '1280px',
      },
      letterSpacing: {
        tightest: '-0.04em',
        tighter: '-0.025em',
      },
      animation: {
        'fade-up': 'fadeUp 600ms cubic-bezier(0.16, 1, 0.3, 1) both',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
