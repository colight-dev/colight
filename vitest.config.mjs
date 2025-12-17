import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: [
      'packages/colight/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      'packages/colight-publish/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'
    ],
    setupFiles: ['packages/colight-publish/tests/js/setup.ts']
  }
})
