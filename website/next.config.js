/** @type {import('next').NextConfig} */
// basePath / assetPrefix only apply when explicitly built for GitHub Pages
// (NEXT_PUBLIC_BASE_PATH set by the deploy workflow). Local `npm run dev` is
// unaffected and serves from /.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig = {
  output: 'export',
  reactStrictMode: true,
  trailingSlash: true,
  images: { unoptimized: true },
  basePath,
  assetPrefix: basePath || undefined,
};

module.exports = nextConfig;
