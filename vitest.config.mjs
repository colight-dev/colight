import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['packages/colight/tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}']
  }
})
