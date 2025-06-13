import esbuild from 'esbuild'
import * as importMap from "esbuild-plugin-import-map";

const args = process.argv.slice(2);
const watch = args.includes('--watch');

// Common options for all builds
const commonOptions = {
  bundle: true,
  plugins: [],
  minify: !watch,
  sourcemap: watch,
  loader: {
    '.css': 'text'
  }
};

// Widget build (ESM)
const widgetConfig = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['src/colight/js/widget.jsx'],
  outfile: 'src/colight/dist/widget.mjs',
  plugins: [],
};

// AnyWidget build (ESM)
const anyWidgetConfig = {
  ...widgetConfig,
  format: 'esm',
  entryPoints: ['src/colight/js/anywidget.jsx'],
  outfile: 'src/colight/dist/anywidget.mjs',
};

// Embed build (IIFE format for standalone use with script tags)
const embedConfigJS = {
  ...commonOptions,
  format: 'iife',
  globalName: 'colight', // Makes it available as window.colight
  entryPoints: ['src/colight/js/embed.js'],
  outfile: 'src/colight/dist/embed.js',
  plugins: [],
};

const embedConfigESM = { ...embedConfigJS, format: 'esm', outfile: 'src/colight/dist/embed.mjs' }

const configs = [widgetConfig, anyWidgetConfig, embedConfigJS, embedConfigESM]

// Apply CDN imports if enabled
const USE_CDN_IMPORTS = false //!watch
if (USE_CDN_IMPORTS) {
  importMap.load('src/colight/js/import-map.cdn.json');
  configs.forEach(config => {
    config.plugins.push(importMap.plugin());
  });
}

async function runBuild() {
  try {

    if (watch) {
      // Watch mode - create contexts for all builds
      const contexts = await Promise.all(
        configs.map(config => esbuild.context(config))
      );

      // Start watching
      await Promise.all(
        contexts.map(context => context.watch())
      );

      console.log('Watching for changes...');
    } else {
      // Build once
      await Promise.all(
        configs.map(config => esbuild.build(config))
      );

      console.log('Build completed successfully');
      console.log('Output files:');
      configs.forEach(config => {
        console.log(' - ' + config.outfile);
      });
    }
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

runBuild();
