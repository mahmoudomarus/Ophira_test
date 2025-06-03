/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    OPHIRA_API_URL: process.env.OPHIRA_API_URL || 'http://localhost:8001',
    OPHIRA_WS_URL: process.env.OPHIRA_WS_URL || 'ws://localhost:8001',
  },
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/ophira/:path*',
        destination: 'http://localhost:8001/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig; 