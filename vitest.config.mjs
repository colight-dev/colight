import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@colight/serde': path.resolve(__dirname, 'packages/colight-serde/src/js/index.ts')
    }
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: [
      'packages/colight/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      'packages/colight-serde/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'
    ],
    setupFiles: ['packages/colight/tests/js/setup.ts']
  }
})
