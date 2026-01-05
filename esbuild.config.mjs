import esbuild from 'esbuild'
import * as importMap from "esbuild-plugin-import-map";

const args = process.argv.slice(2);
const watch = args.includes('--watch');

const DIST_DIR = 'packages/colight/src/colight/js-dist'
const NPM_DIST_DIR = 'packages/colight-scene3d/dist'

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
const widgetESM = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['packages/colight/src/js/widget.jsx'],
  outfile: DIST_DIR+'/widget.mjs',
  plugins: [],
};

// AnyWidget build (ESM)
const anywidgetESM = {
  ...widgetESM,
  format: 'esm',
  entryPoints: ['packages/colight/src/js/anywidget.jsx'],
  outfile: DIST_DIR+'/anywidget.mjs',
};

// Embed build (IIFE format for standalone use with script tags)
const embedConfigJS = {
  ...commonOptions,
  format: 'iife',
  entryPoints: ['packages/colight/src/js/embed.js'],
  outfile: DIST_DIR+'/embed.js',
  plugins: [],
};

// AnyWidget build (ESM)
const embedConfigESM = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['packages/colight/src/js/embed.js'],
  outfile: DIST_DIR+'/embed.mjs',
};

// LiveServer build (IIFE format for embedding in HTML pages)
const liveConfig = {
  ...commonOptions,
  format: 'iife',
  entryPoints: ['packages/colight-prose/src/js/live.jsx'],
  outfile: DIST_DIR+'/live.js',
  plugins: [],
  define: {
    'process.env.NODE_ENV': JSON.stringify(watch ? 'development' : 'production')
  }
};

const scene3dESM = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['packages/colight/src/js/scene3d/index.ts'],
  outfile: NPM_DIST_DIR + '/scene3d.mjs',
  external: ['react'],
  plugins: [],
};

const configs = [
  widgetESM,
  anywidgetESM,
  embedConfigJS,
  embedConfigESM,
  liveConfig,
  scene3dESM,
]

// Apply CDN imports if enabled
const USE_CDN_IMPORTS = false //!watch
if (USE_CDN_IMPORTS) {
  importMap.load('packages/colight/src/js/import-map.cdn.json');
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
